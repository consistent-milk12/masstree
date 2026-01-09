// ========================================================================
//  Compaction
// ========================================================================

use super::{INITIAL_CAPACITY, PermutationProvider, SlotMeta, SuffixBag};

impl<const WIDTH: usize> SuffixBag<WIDTH> {
    /// Compact in-pace by rebuilding the data buffer with only active suffixes.
    ///
    /// This is more efficient than `compact()` when we don't have an external
    /// list of active slots, it just uses the slot metadata directly.
    ///
    /// Uses existing capacity when possible to avoid allocation.
    #[expect(clippy::indexing_slicing, reason = "Bounds checked via slot iteration")]
    pub(super) fn compact_in_place(&mut self) {
        if self.suffix_count == 0 {
            self.data.clear();
            return;
        }

        // Calculate total size of active suffixes
        let new_size: usize = self
            .slots
            .iter()
            .filter(|s: &&SlotMeta| s.has_suffix())
            .map(|s: &SlotMeta| s.len as usize)
            .sum();

        // Allocate new buffer (reuse capacity if sufficient)
        let new_capacity: usize = new_size.next_power_of_two().max(INITIAL_CAPACITY);
        let mut new_data: Vec<u8> = Vec::with_capacity(new_capacity);

        // Copy active suffixes in slot order
        for slot in 0..WIDTH {
            let meta: SlotMeta = self.slots[slot];

            if !meta.has_suffix() {
                continue;
            }

            let start: usize = meta.offset as usize;
            let end: usize = start + meta.len as usize;
            let suffix: &[u8] = &self.data[start..end];

            let new_offset: usize = new_data.len();
            new_data.extend_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation)]
            {
                self.slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: meta.len,
                    _pad: 0,
                };
            }
        }

        self.data = new_data;
        // suffix_count unchanged, we kept all slots that had suffixes
    }

    /// Compact the suffix bag, keeping only the specified active slots.
    ///
    /// This creates a new data buffer containing only the suffixes for
    /// slots that are both marked active AND have suffixes stored.
    /// This effectively garbage-collects unused suffix data.
    ///
    /// Uses stack-allocated scratch space instead of heap allocation for
    /// the active slot list (WIDTH is always small, typically 15 or 24).
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

        // Stack-allocated scratch space for active slots
        let mut active: [usize; WIDTH] = [0; WIDTH];
        let mut active_count: usize = 0;
        let mut new_size: usize = 0;

        // Single pass: collect active slots and calculate new size
        for slot in active_slots {
            if slot >= WIDTH {
                continue;
            }

            let meta: SlotMeta = self.slots[slot];

            if meta.has_suffix() && (active_count < WIDTH) {
                active[active_count] = slot;
                active_count += 1;
                new_size += meta.len as usize;
            }
        }

        if active_count == 0 {
            self.data.clear();
            self.slots = [SlotMeta::EMPTY; WIDTH];
            self.suffix_count = 0;
            return old_used;
        }

        // Allocate new buffer with power-of-2 capacity
        let new_capacity: usize = new_size.next_power_of_two().max(INITIAL_CAPACITY);
        let mut new_data: Vec<u8> = Vec::with_capacity(new_capacity);

        // Reset all slots, then populate with only active ones
        let mut new_slots: [SlotMeta; WIDTH] = [SlotMeta::EMPTY; WIDTH];

        for &slot in &active[..active_count] {
            let meta: SlotMeta = self.slots[slot];

            let start: usize = meta.offset as usize;
            let end: usize = start + meta.len as usize;
            let suffix: &[u8] = &self.data[start..end];

            let new_offset: usize = new_data.len();
            new_data.extend_from_slice(suffix);

            #[expect(clippy::cast_possible_truncation)]
            {
                new_slots[slot] = SlotMeta {
                    offset: new_offset as u32,
                    len: meta.len,
                    _pad: 0,
                };
            }
        }

        self.data = new_data;
        self.slots = new_slots;

        #[expect(clippy::cast_possible_truncation)]
        {
            self.suffix_count = active_count as u8;
        }

        old_used.saturating_sub(self.data.len())
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
