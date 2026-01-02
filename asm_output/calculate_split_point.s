<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point:
.Lfunc_begin174:
	.cfi_startproc
	push r15
	.cfi_def_cfa_offset 16
	push r14
	.cfi_def_cfa_offset 24
	push rbx
	.cfi_def_cfa_offset 32
	.cfi_offset rbx, -32
	.cfi_offset r14, -24
	.cfi_offset r15, -16
	mov r15, rdx
	mov r14, rsi
	mov rbx, rdi
.Ltmp19179:
	lea rdi, [rsi + 64]
.Ltmp19180:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp19181:
	call rax
.Ltmp19182:
	mov r9d, eax
	and r9d, 31
.Ltmp19183:
	cmp r9, 2
	jb .LBB174_1
	mov esi, r9d
	shr esi
.Ltmp19185:
	.p2align	4
.LBB174_3:
	lea r10, [rsi - 1]
.Ltmp19187:
	lea ecx, [r10 + 4*r10]
	add cl, 5
	mov r8, rax
	shrd r8, rdx, cl
	mov rdi, rdx
	shr rdi, cl
	test cl, 64
	cmove rdi, r8
.Ltmp19188:
	lea ecx, [rsi + 4*rsi]
	add cl, 5
	mov r11, rax
	shrd r11, rdx, cl
.Ltmp19189:
	and edi, 31
.Ltmp19190:
	mov r8, rdx
	shr r8, cl
	test cl, 64
	cmove r8, r11
.Ltmp19191:
	cmp rdi, 24
	jae .LBB174_17
.Ltmp19192:
	and r8d, 31
.Ltmp19193:
	mov rcx, qword ptr [r14 + 8*rdi + 128]
.Ltmp19194:
	cmp r8d, 24
	jae .LBB174_18
.Ltmp19195:
	mov rdi, qword ptr [r14 + 8*r8 + 128]
.Ltmp19196:
	cmp rcx, rdi
	jne .LBB174_12
.Ltmp19197:
	cmp r15, rcx
	seta cl
.Ltmp19198:
	sbb cl, 0
.Ltmp19199:
	je .LBB174_9
	movzx ecx, cl
	cmp ecx, 255
	jne .LBB174_12
.Ltmp19201:
	test r10, r10
	jne .LBB174_10
	jmp .LBB174_11
.Ltmp19202:
	.p2align	4
.LBB174_9:
	inc rsi
	mov r10, rsi
.Ltmp19205:
	test r10, r10
	je .LBB174_11
.Ltmp19206:
.LBB174_10:
	mov rsi, r10
	cmp r10, r9
	jb .LBB174_3
	jmp .LBB174_12
.LBB174_11:
	mov rsi, r10
.Ltmp19208:
.LBB174_12:
	test rsi, rsi
	sete cl
	cmp rsi, r9
	setae dil
	or dil, cl
	je .LBB174_14
.Ltmp19209:
.LBB174_1:
	xor eax, eax
.Ltmp19210:
	jmp .LBB174_16
.Ltmp19211:
.LBB174_14:
	lea ecx, [rsi + 4*rsi]
	add cl, 5
	shrd rax, rdx, cl
.Ltmp19212:
	shr rdx, cl
.Ltmp19213:
	test cl, 64
	cmove rdx, rax
	and edx, 31
.Ltmp19214:
	cmp rdx, 24
	jae .LBB174_19
.Ltmp19215:
	mov rax, qword ptr [r14 + 8*rdx + 128]
.Ltmp19216:
	mov qword ptr [rbx + 8], rsi
	mov qword ptr [rbx + 16], rax
	mov eax, 1
.Ltmp19217:
.LBB174_16:
	mov qword ptr [rbx], rax
.Ltmp19218:
	pop rbx
	.cfi_def_cfa_offset 24
	pop r14
.Ltmp19219:
	.cfi_def_cfa_offset 16
	pop r15
.Ltmp19220:
	.cfi_def_cfa_offset 8
	ret
.Ltmp19221:
.LBB174_17:
	.cfi_def_cfa_offset 32
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
.Ltmp19222:
	mov esi, 24
.Ltmp19223:
	call core::panicking::panic_bounds_check
.Ltmp19224:
.LBB174_18:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
.Ltmp19225:
	mov esi, 24
.Ltmp19226:
	mov rdi, r8
.Ltmp19227:
	call core::panicking::panic_bounds_check
.Ltmp19228:
.LBB174_19:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
	mov esi, 24
.Ltmp19229:
	mov rdi, rdx
	mov rdx, rax
.Ltmp19230:
	call core::panicking::panic_bounds_check
.Ltmp19231:
.Lfunc_end174:
	.size	<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point, .Lfunc_end174-<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point
	.cfi_endproc

.section ".text.<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated","ax",@progbits
	.p2align	4
.type	<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated,@function
<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated:
<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point:
.Lfunc_begin215:
	.cfi_startproc
	push rbx
	.cfi_def_cfa_offset 16
	.cfi_offset rbx, -16
	mov rax, qword ptr [rsi + 56]
.Ltmp23458:
	mov r9d, eax
	and r9d, 15
.Ltmp23459:
	cmp r9, 2
	jae .LBB215_5
	xor eax, eax
.Ltmp23461:
	mov qword ptr [rdi], rax
.Ltmp23462:
	pop rbx
	.cfi_def_cfa_offset 8
	ret
.Ltmp23463:
.LBB215_5:
	.cfi_def_cfa_offset 16
	mov r8d, r9d
	shr r8d
.Ltmp23464:
	.p2align	4
.LBB215_6:
	lea r10, [r8 - 1]
.Ltmp23466:
	lea ecx, [4*r10]
	add cl, 4
	mov r11, rax
	shr r11, cl
	and r11d, 15
.Ltmp23467:
	lea ecx, [4*r8]
	add cl, 4
	mov rbx, rax
	shr rbx, cl
.Ltmp23468:
	cmp r11, 15
	je .LBB215_17
.Ltmp23469:
	and ebx, 15
.Ltmp23470:
	mov rcx, qword ptr [rsi + 8*r11 + 64]
.Ltmp23471:
	cmp ebx, 15
	je .LBB215_17
.Ltmp23472:
	mov r11, qword ptr [rsi + 8*rbx + 64]
.Ltmp23473:
	cmp rcx, r11
	jne .LBB215_3
.Ltmp23474:
	cmp rdx, rcx
	seta cl
.Ltmp23475:
	sbb cl, 0
.Ltmp23476:
	je .LBB215_11
	movzx ecx, cl
	cmp ecx, 255
	jne .LBB215_3
.Ltmp23478:
	test r10, r10
	jne .LBB215_2
	jmp .LBB215_13
.Ltmp23479:
	.p2align	4
.LBB215_11:
	inc r8
	mov r10, r8
.Ltmp23482:
	test r10, r10
	je .LBB215_13
.Ltmp23483:
.LBB215_2:
	mov r8, r10
	cmp r10, r9
	jb .LBB215_6
	jmp .LBB215_3
.LBB215_13:
	mov r8, r10
.Ltmp23485:
.LBB215_3:
	test r8, r8
	sete cl
	cmp r8, r9
	setae dl
.Ltmp23486:
	or dl, cl
	je .LBB215_14
.Ltmp23487:
	xor eax, eax
.Ltmp23488:
	mov qword ptr [rdi], rax
.Ltmp23489:
	pop rbx
	.cfi_def_cfa_offset 8
	ret
.Ltmp23490:
.LBB215_14:
	.cfi_def_cfa_offset 16
	lea ecx, [4*r8]
	add cl, 4
	shr rax, cl
.Ltmp23491:
	and eax, 15
.Ltmp23492:
	cmp rax, 15
	je .LBB215_17
.Ltmp23493:
	mov rax, qword ptr [rsi + 8*rax + 64]
.Ltmp23494:
	mov qword ptr [rdi + 8], r8
	mov qword ptr [rdi + 16], rax
	mov eax, 1
.Ltmp23495:
	mov qword ptr [rdi], rax
.Ltmp23496:
	pop rbx
	.cfi_def_cfa_offset 8
	ret
.Ltmp23497:
.LBB215_17:
	.cfi_def_cfa_offset 16
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.183]
	mov edi, 15
	mov esi, 15
.Ltmp23498:
	call core::panicking::panic_bounds_check
.Ltmp23499:
.Lfunc_end215:
	.size	<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point, .Lfunc_end215-<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point
	.cfi_endproc

.section .rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI216_0:
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	1
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
.LCPI216_1:
	.quad	2
	.quad	2
.LCPI216_2:
	.quad	4
	.quad	4
.section ".text.<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated","ax",@progbits
	.p2align	4
.type	<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated,@function
