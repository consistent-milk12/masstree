portable_atomic::imp::atomic128::x86_64::atomic_load::detect:
	.cfi_startproc
	mov eax, dword ptr [rip + portable_atomic::imp::atomic128::x86_64::detect::detect::CACHE.0]
	test eax, eax
	je .LBB294_1
.LBB294_2:
	test al, 4
	lea rcx, [rip + portable_atomic::imp::atomic128::x86_64::atomic_load_cmpxchg16b]
	lea rdx, [rip + portable_atomic::imp::atomic128::x86_64::atomic_load_vmovdqa]
	cmove rdx, rcx
	test al, 2
	lea rax, [rip + portable_atomic::imp::atomic128::x86_64::fallback::atomic_load_seqcst]
	cmovne rax, rdx
	mov qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC], rax
	jmp rax
.LBB294_1:
	push rbx
	.cfi_def_cfa_offset 16
	sub rsp, 16
	.cfi_def_cfa_offset 32
	.cfi_offset rbx, -16
	mov dword ptr [rsp + 12], 1
	lea rax, [rsp + 12]
	mov rbx, rdi
	mov rdi, rax
	call portable_atomic::imp::atomic128::x86_64::detect::_detect
	mov rdi, rbx
	mov eax, dword ptr [rsp + 12]
	mov dword ptr [rip + portable_atomic::imp::atomic128::x86_64::detect::detect::CACHE.0], eax
	add rsp, 16
	.cfi_def_cfa_offset 16
	pop rbx
	.cfi_def_cfa_offset 8
	.cfi_restore rbx
	jmp .LBB294_2
.Lfunc_end294:
	.size	portable_atomic::imp::atomic128::x86_64::atomic_load::detect, .Lfunc_end294-portable_atomic::imp::atomic128::x86_64::atomic_load::detect
	.cfi_endproc

.section .text.portable_atomic::imp::atomic128::x86_64::atomic_load_cmpxchg16b,"ax",@progbits
	.p2align	4
.type	portable_atomic::imp::atomic128::x86_64::atomic_load_cmpxchg16b,@function
portable_atomic::imp::atomic128::x86_64::atomic_load_cmpxchg16b:
portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC:
	.quad	portable_atomic::imp::atomic128::x86_64::atomic_load::detect
	.size	portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC, 8

	.type	portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC,@object
.section .data.portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC,"aw",@progbits
	.p2align	3, 0x0
