//! Tree visualization for `MassTreeGeneric`.
//!
//! Enabled via the `debug-print` feature. All printing uses OCC reads (no locks).
//! The output uses box-drawing characters and ASCII tables for clear structure.

use std::fmt::{self as StdFmt, Formatter, Write as FmtWrite};

#[allow(unused_imports)]
use seize::LocalGuard;

use crate::{
    MassTreeGeneric, Permuter, TreeAllocator,
    internode::InternodeNode,
    leaf15::{
        KSUF_KEYLENX, LAYER_KEYLENX, LeafNode15, MODSTATE_DELETED_LAYER, MODSTATE_EMPTY,
        MODSTATE_INSERT, MODSTATE_REMOVE,
    },
    nodeversion::NodeVersion,
    policy::{LeafPolicy, SlotKind},
};

/// Maximum OCC retry attempts before annotating the node as modified.
const MAX_RETRIES: usize = 3;

/// Recursion limit for sublayer depth to prevent stack overflow.
const MAX_DEPTH: usize = 32;

// ============================================================================
//  Helpers
// ============================================================================

/// Format an ikey (big-endian u64) as hex + printable ASCII.
fn ikey_display(ikey: u64) -> String {
    let bytes: [u8; _] = ikey.to_be_bytes();

    let printable: String = bytes
        .iter()
        .map(|b: &u8| {
            if (*b).is_ascii_graphic() || *b == b' ' {
                char::from(*b)
            } else {
                '.'
            }
        })
        .collect();

    format!("{ikey:016x} \"{printable}\"")
}

/// Format a byte slice as printable ASCII, escaping non-printable bytes.
fn bytes_display(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());

    for &b in bytes {
        if b.is_ascii_graphic() || b == b' ' {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "\\x{b:02x}");
        }
    }

    out
}

/// Format a raw pointer as hex or "null".
fn ptr_str<T>(ptr: *mut T) -> String {
    if ptr.is_null() {
        "null".to_owned()
    } else {
        format!("{ptr:p}")
    }
}

/// Build a table row separator.
/// `joints` specifies the box-drawing chars: (left, cross, right) and `dash`.
fn table_rule(widths: &[usize], left: char, cross: char, right: char, dash: char) -> String {
    let mut s = String::new();
    s.push(left);

    for (i, &w) in widths.iter().enumerate() {
        for _ in 0..w {
            s.push(dash);
        }

        if i + 1 < widths.len() {
            s.push(cross);
        }
    }

    s.push(right);
    s
}

// ============================================================================
//  Printing Context
// ============================================================================

/// Carries indentation state through recursive calls.
struct Ctx<'w, W: FmtWrite> {
    out: &'w mut W,

    /// Current sublayer depth (0 = top-level tree).
    depth: usize,

    /// Prefix string for the current line (e.g., "|   |   ").
    prefix: String,
}

impl<W: FmtWrite> Ctx<'_, W> {
    fn writeln_prefixed(&mut self, text: &str) -> StdFmt::Result {
        writeln!(self.out, "{}{text}", self.prefix)
    }
}

// ============================================================================
//  Node Dispatch
// ============================================================================

/// Climb parent pointers from a layer-slot target to the true layer root.
///
/// After promotion the slot still points at the original left leaf; the root
/// is the parentless ancestor that owns the whole sublayer chain.
///
/// SAFETY: `node_ptr` must be a live leaf or internode protected by `guard`.
unsafe fn resolve_layer_root<P: LeafPolicy>(
    mut node_ptr: *mut u8,
    guard: &LocalGuard<'_>,
) -> *mut u8 {
    for _ in 0..64 {
        if node_ptr.is_null() {
            return node_ptr;
        }

        // SAFETY: Both node types have NodeVersion as their first field.
        let version: &NodeVersion = unsafe { &*node_ptr.cast::<NodeVersion>() };

        let parent: *mut u8 = if version.is_leaf() {
            // SAFETY: is_leaf() confirmed.
            unsafe { (*node_ptr.cast::<LeafNode15<P>>()).parent(guard) }
        } else {
            // SAFETY: !is_leaf() confirmed.
            unsafe { (*node_ptr.cast::<InternodeNode>()).parent(guard) }
        };

        if parent.is_null() {
            return node_ptr;
        }

        node_ptr = parent;
    }

    node_ptr
}

/// Dispatch a raw node pointer to the correct print function.
///
/// SAFETY: `ptr` must be non-null and point to a live `LeafNode15<P>` or
/// `InternodeNode` that is EBR-protected by the caller's guard.
unsafe fn print_node<P, W>(
    ctx: &mut Ctx<'_, W>,
    ptr: *mut u8,
    guard: &LocalGuard<'_>,
) -> StdFmt::Result
where
    P: LeafPolicy,
    P::Value: StdFmt::Debug,
    W: FmtWrite,
{
    if ptr.is_null() {
        return ctx.writeln_prefixed("[null]");
    }

    // SAFETY: Both node types have NodeVersion as their first field.
    let version: &NodeVersion = unsafe { &*ptr.cast::<NodeVersion>() };

    if version.is_leaf() {
        // SAFETY: is_leaf() confirmed.
        unsafe { print_leaf::<P, W>(ctx, ptr.cast::<LeafNode15<P>>(), guard) }
    } else {
        // SAFETY: !is_leaf() confirmed.
        unsafe { print_internode::<P, W>(ctx, ptr, guard) }
    }
}

// ============================================================================
//  Leaf Printing
// ============================================================================

// Table column widths (including padding).
const COL_POS: usize = 5;
const COL_IKEY: usize = 30;
const COL_KLX: usize = 11;
const COL_SLOT: usize = 6;
const COL_VAL: usize = 30;

/// Render the slot table for a leaf and collect layer pointers for sublayer recursion.
///
/// SAFETY: `leaf` must be a valid reference to an EBR-protected leaf.
unsafe fn print_leaf_slots<P, W>(
    ctx: &mut Ctx<'_, W>,
    leaf: &LeafNode15<P>,
    perm: crate::permuter::Permuter15,
    size: usize,
) -> Result<Vec<(usize, *mut u8)>, StdFmt::Error>
where
    P: LeafPolicy,
    P::Value: StdFmt::Debug,
    W: FmtWrite,
{
    let widths: [usize; 5] = [COL_POS, COL_IKEY, COL_KLX, COL_SLOT, COL_VAL];
    let top_rule: String = table_rule(&widths, '\u{250c}', '\u{252c}', '\u{2510}', '\u{2500}');
    let mid_rule: String = table_rule(&widths, '\u{251c}', '\u{253c}', '\u{2524}', '\u{2500}');
    let bot_rule: String = table_rule(&widths, '\u{2514}', '\u{2534}', '\u{2518}', '\u{2500}');

    ctx.writeln_prefixed(&top_rule)?;
    ctx.writeln_prefixed(
        "\u{2502} pos \u{2502} ikey                         \u{2502} keylenx   \u{2502} slot \u{2502} value                        \u{2502}",
    )?;
    ctx.writeln_prefixed(&mid_rule)?;

    let mut layer_ptrs: Vec<(usize, *mut u8)> = Vec::new();

    for pos in 0..size {
        let slot: usize = perm.get(pos);
        let ikey: u64 = leaf.ikey(slot);
        let klx: u8 = leaf.keylenx(slot);
        let ikey_s: String = ikey_display(ikey);

        let klx_str: String = if klx >= LAYER_KEYLENX {
            format!("{klx:>3} LAYER")
        } else if klx == KSUF_KEYLENX {
            format!("{klx:>3} KSUF ")
        } else {
            format!("{klx:>3}      ")
        };

        let value_str = match leaf.classify_slot(slot) {
            SlotKind::Empty => "<empty>".to_owned(),

            SlotKind::Value(v) => {
                // SAFETY: `leaf` is EBR-protected by the caller's guard. The
                // leaf version is validated after the whole table is rendered,
                // so this is the standard read-then-validate OCC order.
                // `output_to_value_ref` returns an atomic copy for write-through
                // values, so `Debug` never aliases a concurrently written Box.
                let value_ref = unsafe { P::output_to_value_ref(&v) };
                format!("{:?}", &*value_ref)
            }

            SlotKind::Layer(ptr) => {
                layer_ptrs.push((pos, ptr));
                format!("-> sublayer @ {}", ptr_str(ptr))
            }
        };

        let suffix_str: String = if leaf.has_ksuf(slot) {
            leaf.ksuf(slot)
                .map(|s: &[u8]| format!(" sfx=\"{}\"", bytes_display(s)))
                .unwrap_or_default()
        } else {
            String::new()
        };

        let val_col: String = format!("{value_str}{suffix_str}");

        ctx.writeln_prefixed(&format!(
            "\u{2502} {pos:>3} \u{2502} {ikey_s:<28} \u{2502} {klx_str} \u{2502} {slot:>4} \u{2502} {val_col:<29}\u{2502}"
        ))?;
    }

    ctx.writeln_prefixed(&bot_rule)?;
    Ok(layer_ptrs)
}

/// Print a leaf node with OCC-safe field reads and an ASCII slot table.
///
/// SAFETY: `leaf_ptr` must point to a live `LeafNode15<P>` protected by EBR.
unsafe fn print_leaf<P, W>(
    ctx: &mut Ctx<'_, W>,
    leaf_ptr: *mut LeafNode15<P>,
    guard: &LocalGuard<'_>,
) -> StdFmt::Result
where
    P: LeafPolicy,
    P::Value: StdFmt::Debug,
    W: FmtWrite,
{
    // SAFETY: caller guarantees leaf_ptr is valid and EBR-protected.
    let leaf: &LeafNode15<P> = unsafe { &*leaf_ptr };

    for attempt in 0..MAX_RETRIES {
        let vsn: u32 = leaf.version().stable();

        // Snapshot all fields
        let deleted: bool = leaf.version().is_deleted();
        let perm: Permuter = leaf.permutation();
        let size: usize = perm.size();
        let modstate: u8 = leaf.modstate();
        let bound_ikey: u64 = leaf.ikey_bound();
        // SAFETY: unguarded reads for display only (EBR-protected).
        let prev_ptr: *mut LeafNode15<P> = unsafe { leaf.prev_unguarded() };
        let next_ptr: *mut LeafNode15<P> = unsafe { leaf.safe_next_unguarded() };

        // OCC validation
        if leaf.version().has_changed(vsn) {
            if attempt + 1 == MAX_RETRIES {
                return ctx
                    .writeln_prefixed(&format!("Leaf [CONCURRENT MODIFICATION] @ {leaf_ptr:p}"));
            }

            continue;
        }

        // Header tags
        let del_tag: &str = if deleted { " [DELETED]" } else { "" };
        let mod_tag: &str = match modstate & 0x03 {
            MODSTATE_INSERT => "",
            MODSTATE_REMOVE => " [REMOVING]",
            MODSTATE_DELETED_LAYER => " [DELETED_LAYER]",
            MODSTATE_EMPTY => " [EMPTY]",
            _ => " [?]",
        };

        // Render the entire leaf section into a private buffer so the slot
        // reads can be validated *after* the last one. Writing directly to
        // `ctx` would expose a torn table on a retry.
        let mut leaf_buf: String = String::new();
        let layer_ptrs: Vec<(usize, *mut u8)>;
        {
            let mut local = Ctx {
                out: &mut leaf_buf,
                depth: ctx.depth,
                prefix: ctx.prefix.clone(),
            };

            local.writeln_prefixed(&format!(
                "Leaf  {size}/15  v=0x{vsn:04x}  bound={}  @ {leaf_ptr:p}{del_tag}{mod_tag}",
                ikey_display(bound_ikey),
            ))?;

            local.writeln_prefixed(&format!(
                "chain: {} <- {:p} -> {}",
                ptr_str(prev_ptr),
                leaf_ptr,
                ptr_str(next_ptr),
            ))?;

            if size == 0 {
                local.writeln_prefixed("(empty)")?;
                // SAFETY: leaf is valid and EBR-protected (same as above).
                if leaf.version().has_changed(vsn) {
                    if attempt + 1 == MAX_RETRIES {
                        return ctx.writeln_prefixed(&format!(
                            "Leaf [CONCURRENT MODIFICATION] @ {leaf_ptr:p}"
                        ));
                    }
                    continue;
                }
                ctx.out.write_str(&leaf_buf)?;
                return Ok(());
            }

            let perm_slots: Vec<usize> = (0..size).map(|i: usize| perm.get(i)).collect();
            local.writeln_prefixed(&format!("perm: {perm_slots:?}"))?;

            // SAFETY: leaf is valid and EBR-protected (same as above).
            layer_ptrs = unsafe { print_leaf_slots::<P, String>(&mut local, leaf, perm, size)? };
        }

        // Validate after the complete slot table has been read. On a change,
        // retry from a fresh version and discard the buffered section.
        if leaf.version().has_changed(vsn) {
            if attempt + 1 == MAX_RETRIES {
                return ctx
                    .writeln_prefixed(&format!("Leaf [CONCURRENT MODIFICATION] @ {leaf_ptr:p}"));
            }

            continue;
        }

        ctx.out.write_str(&leaf_buf)?;

        // Recurse into sublayers. After layer-root promotion the parent slot
        // still points at the original left leaf; climb to the true root so
        // the internode and every sibling leaf are printed.
        for (pos, layer_ptr) in layer_ptrs {
            if !layer_ptr.is_null() && ctx.depth < MAX_DEPTH {
                // SAFETY: layer_ptr is EBR-protected and resolve_layer_root
                // only follows parent pointers of live nodes under the guard.
                let layer_root: *mut u8 = unsafe { resolve_layer_root::<P>(layer_ptr, guard) };

                ctx.writeln_prefixed(&format!(
                    "\u{2514}\u{2500}\u{2500} SUBLAYER (slot pos={pos}, root @ {layer_root:p})"
                ))?;

                let saved: String = ctx.prefix.clone();
                ctx.prefix.push_str("    ");
                ctx.depth += 1;

                // SAFETY: layer_root is the true sublayer root, EBR-protected.
                unsafe { print_node::<P, W>(ctx, layer_root, guard)? };

                ctx.depth -= 1;
                ctx.prefix = saved;
            }
        }

        return Ok(());
    }
    Ok(())
}

// ============================================================================
//  Internode Printing
// ============================================================================

/// Print an internode with OCC-safe reads, recursing into children.
///
/// SAFETY: `inode_ptr` must point to a live `InternodeNode`, EBR-protected.
unsafe fn print_internode<P, W>(
    ctx: &mut Ctx<'_, W>,
    inode_ptr: *mut u8,
    guard: &LocalGuard<'_>,
) -> StdFmt::Result
where
    P: LeafPolicy,
    P::Value: StdFmt::Debug,
    W: FmtWrite,
{
    // SAFETY: caller guarantees ptr is valid InternodeNode, EBR-protected.
    let inode: &InternodeNode = unsafe { &*inode_ptr.cast::<InternodeNode>() };

    for attempt in 0..MAX_RETRIES {
        let vsn: u32 = inode.version().stable();
        let deleted: bool = inode.version().is_deleted();
        let nkeys: usize = inode.nkeys();
        let height: u32 = inode.height();

        // Snapshot separator keys
        let mut seps: Vec<u64> = Vec::with_capacity(nkeys);

        for i in 0..nkeys {
            seps.push(inode.ikey(i));
        }

        // Snapshot child pointers
        let mut children: Vec<*mut u8> = Vec::with_capacity(nkeys + 1);
        for i in 0..=nkeys {
            // SAFETY: child_unguarded under OCC, re-validated below.
            children.push(unsafe { inode.child_unguarded(i) });
        }

        // OCC validation
        if inode.version().has_changed(vsn) {
            if attempt + 1 == MAX_RETRIES {
                return ctx.writeln_prefixed(&format!(
                    "Internode [CONCURRENT MODIFICATION] @ {inode_ptr:p}"
                ));
            }

            continue;
        }

        // Header
        let del_tag: &str = if deleted { " [DELETED]" } else { "" };
        ctx.writeln_prefixed(&format!(
            "Internode  h={height}  nkeys={nkeys}  v=0x{vsn:04x}  @ {inode_ptr:p}{del_tag}"
        ))?;

        // Print children interleaved with separator keys
        for i in 0..=nkeys {
            let child_ptr: *mut u8 = children[i];
            let is_last: bool = i == nkeys;
            let branch: &str = if is_last {
                "\u{2514}\u{2500}\u{2500}"
            } else {
                "\u{251c}\u{2500}\u{2500}"
            };
            let cont: &str = if is_last { "    " } else { "\u{2502}   " };

            // Print separator before this child (except child 0)
            if i > 0 {
                let sep: u64 = seps[i - 1];

                ctx.writeln_prefixed(&format!(
                    "\u{2502}   sep[{}] = {}",
                    i - 1,
                    ikey_display(sep),
                ))?;
            }

            ctx.writeln_prefixed(&format!("{branch} [child {i}] @ {}", ptr_str(child_ptr)))?;

            if !child_ptr.is_null() {
                let saved: String = ctx.prefix.clone();
                ctx.prefix.push_str(cont);

                // SAFETY: child_ptr is EBR-protected.
                unsafe { print_node::<P, W>(ctx, child_ptr, guard)? };

                ctx.prefix = saved;
            }
        }

        return Ok(());
    }
    Ok(())
}

// ============================================================================
//  Public API
// ============================================================================

impl<P, A> MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    P::Value: StdFmt::Debug,
    A: TreeAllocator<P>,
{
    /// Write a human-readable tree visualization to `w`.
    ///
    /// Uses OCC reads: never blocks writers. May annotate nodes with
    /// `[CONCURRENT MODIFICATION]` if a node keeps changing during the snapshot.
    ///
    /// # Errors
    ///
    /// Returns `fmt::Error` if writing to `w` fails.
    pub fn print_tree_to(&self, w: &mut impl FmtWrite) -> StdFmt::Result {
        let guard: LocalGuard<'_> = self.guard();
        // SAFETY: the guard keeps the root live for the duration of the print,
        // and this function only reads node contents.
        let root: *mut u8 = unsafe { self.debug_root_ptr(&guard) };
        let count: usize = self.len();

        writeln!(w, "MassTree  count={count}  @ {root:p}")?;

        if root.is_null() {
            return writeln!(w, "\u{2514}\u{2500}\u{2500} [empty]");
        }

        // Type: Ctx<'_, impl Write>
        let mut ctx = Ctx {
            out: w,
            depth: 0,
            prefix: String::new(),
        };

        // SAFETY: root is EBR-protected by `guard`. Both node types have
        // NodeVersion as first field, so the is_leaf() dispatch is sound.
        unsafe { print_node::<P, _>(&mut ctx, root, &guard) }
    }

    /// Return the tree visualization as a `String`.
    pub fn print_tree(&self) -> String {
        let mut s = String::with_capacity(4096);

        // write! to String is infallible.
        let _ = self.print_tree_to(&mut s);

        s
    }
}

impl<P, A> StdFmt::Display for MassTreeGeneric<P, A>
where
    P: LeafPolicy,
    P::Value: StdFmt::Debug,
    A: TreeAllocator<P>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> StdFmt::Result {
        self.print_tree_to(f)
    }
}
