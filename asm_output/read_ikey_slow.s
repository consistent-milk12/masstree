masstree::key::Key::read_ikey_slow:
	.cfi_startproc
	push rax
	.cfi_def_cfa_offset 16
	mov rax, rsi
	mov qword ptr [rsp], 0
	cmp rsi, 9
	jae .LBB0_2
	mov rsi, rdi
	mov rdi, rsp
	mov rdx, rax
	call qword ptr [rip + memcpy@GOTPCREL]
	mov rax, qword ptr [rsp]
	bswap rax
	pop rcx
	.cfi_def_cfa_offset 8
	ret
.LBB0_2:
	.cfi_def_cfa_offset 16
	lea rcx, [rip + .Lanon.74f41b5b7b769a956ebd29afaba4cf35.19]
	mov edx, 8
	xor edi, edi
	mov rsi, rax
	call core::slice::index::slice_index_fail
.Lfunc_end0:
	.size	masstree::key::Key::read_ikey_slow, .Lfunc_end0-masstree::key::Key::read_ikey_slow
	.cfi_endproc

.section ".text.unlikely.alloc::raw_vec::RawVecInner<A>::reserve::do_reserve_and_handle","ax",@progbits
.type	alloc::raw_vec::RawVecInner<A>::reserve::do_reserve_and_handle,@function
