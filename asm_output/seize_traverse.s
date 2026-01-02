seize::raw::collector::Collector::traverse:
	.cfi_startproc
	test rsi, rsi
	je .LBB369_11
	push rbp
	.cfi_def_cfa_offset 16
	push r15
	.cfi_def_cfa_offset 24
	push r14
	.cfi_def_cfa_offset 32
	push r13
	.cfi_def_cfa_offset 40
	push r12
	.cfi_def_cfa_offset 48
	push rbx
	.cfi_def_cfa_offset 56
	push rax
	.cfi_def_cfa_offset 64
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	.cfi_offset rbp, -16
	mov r14, rsi
	mov rbx, rdi
	mov r12, qword ptr [rip + mi_free@GOTPCREL]
	jmp .LBB369_2
	.p2align	4
.LBB369_8:
	mov rdi, r15
	call r12
.LBB369_9:
	test r14, r14
	je .LBB369_10
.LBB369_2:
	mov r15, qword ptr [r14 + 24]
	mov r14, qword ptr [r14 + 16]
	lock dec	qword ptr [r15 + 24]
	jne .LBB369_9
	#MEMBARRIER
	mov r13, qword ptr [r15 + 16]
	test r13, r13
	je .LBB369_6
	mov rax, qword ptr [r15 + 8]
	shl r13, 5
	add r13, rax
	.p2align	4
.LBB369_5:
	lea rbp, [rax + 32]
	mov rdi, qword ptr [rax + 8]
	mov rsi, rbx
	call qword ptr [rax]
	mov rax, rbp
	cmp rbp, r13
	jne .LBB369_5
.LBB369_6:
	cmp qword ptr [r15], 0
	je .LBB369_8
	mov rdi, qword ptr [r15 + 8]
	call r12
	jmp .LBB369_8
.LBB369_10:
	add rsp, 8
	.cfi_def_cfa_offset 56
	pop rbx
	.cfi_def_cfa_offset 48
	pop r12
	.cfi_def_cfa_offset 40
	pop r13
	.cfi_def_cfa_offset 32
	pop r14
	.cfi_def_cfa_offset 24
	pop r15
	.cfi_def_cfa_offset 16
	pop rbp
	.cfi_def_cfa_offset 8
	.cfi_restore rbx
	.cfi_restore r12
	.cfi_restore r13
	.cfi_restore r14
	.cfi_restore r15
	.cfi_restore rbp
.LBB369_11:
	ret
.Lfunc_end369:
	.size	seize::raw::collector::Collector::traverse, .Lfunc_end369-seize::raw::collector::Collector::traverse
	.cfi_endproc

.section .text.seize::collector::Collector::new,"ax",@progbits
	.p2align	4
.type	seize::collector::Collector::new,@function
