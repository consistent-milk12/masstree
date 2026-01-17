// ========================================================================
//  Compaction
// ========================================================================

use arrayvec::ArrayVec;

use super::{PermutationProvider, SlotMeta, SuffixBag};

/// Entry for tracking suffix during compaction.
/// Stores (slot_index, original_offset, length).
#[derive(Clone, Copy)]
struct CompactEntry {
    slot: usize,
    offset: u32,
    len: u16,
}

impl<const WIDTH: usize> SuffixBag<WIDTH> {
    /// Compact in-place by moving data within the existing buffer.
    ///
    /// This is a true in-place compaction that:
    /// 1. Uses stack-allocated ArrayVec for bookkeeping (no heap allocation)
    /// 2. Moves data using `copy_within` (memmove, no allocation)
    /// 3. Truncates the buffer (no allocation)
    ///
    /// # Algorithm
    ///
    /// 1. Collect active suffixes into ArrayVec, sorted by offset
    /// 2. Move each suffix to its new position using copy_within
    /// 3. Update slot metadata with new offsets
    /// 4. Truncate buffer to new size
    #[expect(clippy::indexing_slicing, reason = "Bounds checked via slot iteration")]
    pub(super) fn compact_in_place(&mut self) {
        if self.suffix_count == 0 {
            self.data.clear();
            return;
        }

        // Collect active suffixes into stack-allocated array
        let mut entries: ArrayVec<CompactEntry, WIDTH> = ArrayVec::new();

        for slot in 0..WIDTH {
            let meta: SlotMeta = self.slots[slot];
            if meta.has_suffix() {
                entries.push(CompactEntry {
                    slot,
                    offset: meta.offset,
                    len: meta.len,
                });
            }
        }

        if entries.is_empty() {
            self.data.clear();
            self.suffix_count = 0;
            return;
        }

        // Sort by offset so we can move data in-order without overwriting
        entries.sort_unstable_by_key(|e: &CompactEntry| e.offset);

        // Move each suffix to its compacted position
        let mut write_pos: usize = 0;

        for entry in &entries {
            let src_start: usize = entry.offset as usize;
            let src_end: usize = src_start + entry.len as usize;

            // Only copy if source and dest differ (avoid no-op copies)
            if write_pos != src_start {
                self.data.copy_within(src_start..src_end, write_pos);
            }

            // Update slot metadata with new offset
            #[expect(clippy::cast_possible_truncation)]
            {
                self.slots[entry.slot] = SlotMeta {
                    offset: write_pos as u32,
                    len: entry.len,
                    _pad: 0,
                };
            }

            write_pos += entry.len as usize;
        }

        // Truncate buffer to new size (no allocation!)
        self.data.truncate(write_pos);
        // suffix_count unchanged - we kept all slots that had suffixes
    }

    /// Compact keeping only specified active slots, using in-place moves.
    ///
    /// This is a zero-allocation compaction that:
    /// 1. Uses stack-allocated scratch space for bookkeeping
    /// 2. Moves data using `copy_within` (memmove)
    /// 3. Truncates the buffer
    ///
    /// # Arguments
    ///
    /// * `active_slots` - Iterator yielding physical slot indices that are active
    ///
    /// # Returns
    ///
    /// The number of bytes reclaimed.
    #[expect(clippy::indexing_slicing, reason = "Slot bounds explicitly checked")]
    pub fn compact(&mut self, active_slots: impl Iterator<Item = usize>) -> usize {
        let old_used: usize = self.data.len();

        // Collect active entries into stack-allocated array
        let mut entries: ArrayVec<CompactEntry, WIDTH> = ArrayVec::new();

        for slot in active_slots {
            if slot >= WIDTH {
                continue;
            }

            let meta: SlotMeta = self.slots[slot];
            if meta.has_suffix() {
                entries.push(CompactEntry {
                    slot,
                    offset: meta.offset,
                    len: meta.len,
                });
            }
        }

        if entries.is_empty() {
            self.data.clear();
            self.slots = [SlotMeta::EMPTY; WIDTH];
            self.suffix_count = 0;
            return old_used;
        }

        // Sort by offset for in-order movement
        entries.sort_unstable_by_key(|e: &CompactEntry| e.offset);

        // Reset slots that will become inactive
        let mut new_slots: [SlotMeta; WIDTH] = [SlotMeta::EMPTY; WIDTH];

        // Move each suffix to its compacted position
        let mut write_pos: usize = 0;

        for entry in &entries {
            let src_start: usize = entry.offset as usize;
            let src_end: usize = src_start + entry.len as usize;

            if write_pos != src_start {
                self.data.copy_within(src_start..src_end, write_pos);
            }

            #[expect(clippy::cast_possible_truncation)]
            {
                new_slots[entry.slot] = SlotMeta {
                    offset: write_pos as u32,
                    len: entry.len,
                    _pad: 0,
                };
            }

            write_pos += entry.len as usize;
        }

        // Truncate and update state
        self.data.truncate(write_pos);
        self.slots = new_slots;

        #[expect(clippy::cast_possible_truncation)]
        {
            self.suffix_count = entries.len() as u8;
        }

        old_used.saturating_sub(write_pos)
    }

    /// Compact using a permutation to determine active slots.
    ///
    /// This is the typical usage pattern: compact based on which slots
    /// are currently in-use according to the leaf's permutation.
    ///
    /// # Arguments
    ///
    /// * `perm` - Permuter indicating which slots are active
    /// * `exclude_slot` - Optional slot to exclude (e.g., slot being removed)
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
