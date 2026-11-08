// ========================================================================
//  Compaction
// ========================================================================

use std::sync::atomic::Ordering;

use arrayvec::ArrayVec;

use super::{PermutationProvider, SlotMeta, SuffixBag, WIDTH};

/// Entry for tracking suffix during compaction.
#[derive(Clone, Copy)]
struct CompactEntry {
    slot: usize,
    offset: u32,
    len: u16,
}

impl SuffixBag {
    /// Compact in-place by moving data within the existing buffer.
    ///
    /// Only reachable while the bag is exclusively owned: `try_assign` on an
    /// unpublished bag, or the drain-and-rebuild path while the leaf lock is
    /// held and readers validate with OCC. Not used for append-only concurrent
    /// assignment.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive ownership of the bag; this moves bytes and
    /// invalidates shared slices.
    pub(super) unsafe fn compact_in_place_locked(&self) {
        if self.count() == 0 {
            let data_ptr: *mut Vec<u8> = self.data.get();
            // SAFETY: Caller has exclusive write access.
            let data: &mut Vec<u8> = unsafe { &mut *data_ptr };
            data.clear();
            self.publish_used(0);
            return;
        }

        let mut entries: ArrayVec<CompactEntry, WIDTH> = ArrayVec::new();
        let mut already_sorted: bool = true;
        let mut last_offset: u32 = 0;

        for slot in 0..WIDTH {
            let meta: SlotMeta = self.load_meta(slot);

            if meta.has_suffix() {
                if meta.offset < last_offset {
                    already_sorted = false;
                }
                last_offset = meta.offset;

                entries.push(CompactEntry {
                    slot,
                    offset: meta.offset,
                    len: meta.len,
                });
            }
        }

        if entries.is_empty() {
            let data_ptr: *mut Vec<u8> = self.data.get();
            // SAFETY: Caller has exclusive write access.
            let data: &mut Vec<u8> = unsafe { &mut *data_ptr };
            data.clear();
            self.publish_used(0);
            self.suffix_count.store(0, Ordering::Release);
            return;
        }

        if !already_sorted {
            entries.sort_unstable_by_key(|e: &CompactEntry| e.offset);
        }

        let data_ptr: *mut Vec<u8> = self.data.get();
        // SAFETY: Caller has exclusive write access.
        let data: &mut Vec<u8> = unsafe { &mut *data_ptr };
        let mut write_pos: usize = 0;

        for entry in &entries {
            let src_start: usize = entry.offset as usize;
            let src_end: usize = src_start + entry.len as usize;

            // A zero-length suffix shares offset 0 with whichever suffix was
            // assigned first. Repoint it without consuming or copying bytes.
            if entry.len == 0 {
                #[expect(clippy::cast_possible_truncation)]
                {
                    self.store_meta(
                        entry.slot,
                        SlotMeta {
                            offset: write_pos as u32,
                            len: 0,
                            pad: 0,
                        },
                    );
                }
                continue;
            }

            debug_assert!(
                write_pos <= src_start,
                "compaction invariant violated: write_pos ({write_pos}) > src_start ({src_start})"
            );

            if write_pos != src_start {
                data.copy_within(src_start..src_end, write_pos);
            }

            #[expect(clippy::cast_possible_truncation)]
            {
                self.store_meta(
                    entry.slot,
                    SlotMeta {
                        offset: write_pos as u32,
                        len: entry.len,
                        pad: 0,
                    },
                );
            }

            write_pos += entry.len as usize;
        }

        data.truncate(write_pos);
        self.publish_used(write_pos);
    }

    /// Compact keeping only specified active slots, using in-place moves.
    pub(super) fn compact(&mut self, active_slots: impl Iterator<Item = usize>) -> usize {
        // SAFETY: `&mut self` guarantees exclusive ownership.
        unsafe { self.compact_locked(active_slots) }
    }

    /// Compact keeping only the given active slots.
    ///
    /// # Safety
    ///
    /// Caller must have exclusive ownership of the bag.
    #[expect(clippy::indexing_slicing, reason = "Slot bounds explicitly checked")]
    #[expect(
        clippy::too_many_lines,
        reason = "Compaction keeps allocation and metadata updates together"
    )]
    pub(super) unsafe fn compact_locked(&self, active_slots: impl Iterator<Item = usize>) -> usize {
        let old_used: usize = self.used();
        let mut entries: ArrayVec<CompactEntry, WIDTH> = ArrayVec::new();
        let mut seen: u64 = 0;
        let mut already_sorted: bool = true;
        let mut last_offset: u32 = 0;

        for slot in active_slots {
            debug_assert!(
                slot < WIDTH,
                "active_slots yielded out-of-bounds slot {slot} (WIDTH={WIDTH})"
            );

            if slot >= WIDTH {
                continue;
            }

            let mask: u64 = 1 << slot;

            if seen & mask != 0 {
                continue;
            }

            seen |= mask;

            let meta: SlotMeta = self.load_meta(slot);

            if meta.has_suffix() {
                if meta.offset < last_offset {
                    already_sorted = false;
                }
                last_offset = meta.offset;

                entries.push(CompactEntry {
                    slot,
                    offset: meta.offset,
                    len: meta.len,
                });
            }
        }

        if entries.is_empty() {
            let data_ptr: *mut Vec<u8> = self.data.get();
            // SAFETY: Caller has exclusive write access.
            let data: &mut Vec<u8> = unsafe { &mut *data_ptr };
            data.clear();
            self.publish_used(0);

            for slot in 0..WIDTH {
                self.store_meta(slot, SlotMeta::EMPTY);
            }
            self.suffix_count.store(0, Ordering::Release);
            return old_used;
        }

        if !already_sorted {
            entries.sort_unstable_by_key(|e: &CompactEntry| e.offset);
        }

        let mut new_slots: [SlotMeta; WIDTH] = [SlotMeta::EMPTY; WIDTH];

        let data_ptr: *mut Vec<u8> = self.data.get();
        // SAFETY: Caller has exclusive write access.
        let data: &mut Vec<u8> = unsafe { &mut *data_ptr };
        let mut write_pos: usize = 0;

        for entry in &entries {
            let src_start: usize = entry.offset as usize;
            let src_end: usize = src_start + entry.len as usize;

            // Zero-length suffixes are repointed but never copied; they share
            // offset 0 with the first assigned suffix.
            if entry.len == 0 {
                #[expect(clippy::cast_possible_truncation)]
                {
                    new_slots[entry.slot] = SlotMeta {
                        offset: write_pos as u32,
                        len: 0,
                        pad: 0,
                    };
                }
                continue;
            }

            debug_assert!(
                write_pos <= src_start,
                "compaction invariant violated: write_pos ({write_pos}) > src_start ({src_start})"
            );

            if write_pos != src_start {
                data.copy_within(src_start..src_end, write_pos);
            }

            #[expect(clippy::cast_possible_truncation)]
            {
                new_slots[entry.slot] = SlotMeta {
                    offset: write_pos as u32,
                    len: entry.len,
                    pad: 0,
                };
            }

            write_pos += entry.len as usize;
        }

        data.truncate(write_pos);
        self.publish_used(write_pos);

        for (slot, meta) in new_slots.iter().enumerate() {
            self.store_meta(slot, *meta);
        }

        #[expect(clippy::cast_possible_truncation)]
        {
            self.suffix_count
                .store(entries.len() as u8, Ordering::Release);
        }

        old_used.saturating_sub(write_pos)
    }

    /// Compact using a permutation to determine active slots.
    ///
    /// # Returns
    ///
    /// The number of bytes reclaimed.
    pub fn compact_with_permuter<P: PermutationProvider>(
        &mut self,
        perm: &P,
        exclude_slot: Option<usize>,
    ) -> usize {
        let active = (0..perm.size())
            .map(|i: usize| perm.get(i))
            .filter(|s: &usize| Some(*s) != exclude_slot);

        self.compact(active)
    }
}
