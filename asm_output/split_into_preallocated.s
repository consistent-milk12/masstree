<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated:
.Lfunc_begin175:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception58
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
	sub rsp, 72
	.cfi_def_cfa_offset 128
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	.cfi_offset rbp, -16
	mov qword ptr [rsp + 48], r8
.Ltmp19248:
	mov rbx, rcx
	mov r13, rdx
	mov r15, rsi
	mov r14, rdi
	mov eax, dword ptr [rsi]
	and eax, -2147483648
	or eax, 5
.Ltmp19250:
	mov dword ptr [rcx], eax
.Ltmp19251:
	lea rdi, [rsi + 64]
.Ltmp19252:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp19253:
.Ltmp19232:
	mov qword ptr [rsp + 64], rdi
.Ltmp19254:
	call rax
.Ltmp19255:
	mov qword ptr [rsp + 56], rdx
.Ltmp19256:
.Ltmp19233:
	mov qword ptr [rsp + 24], rax
.Ltmp19257:
	mov r12, rax
.Ltmp19258:
	and r12d, 31
	xor edi, edi
.Ltmp19259:
	mov rax, r12
	mov esi, 0
	sub rax, r13
.Ltmp19261:
	mov qword ptr [rsp + 8], rax
.Ltmp19262:
	jne .LBB175_2
.Ltmp19263:
.LBB175_17:
	mov ecx, r13d
	sub ecx, dword ptr [rsp + 24]
	lea rax, [r13 + 23]
	test cl, 1
	jne .LBB175_19
	cmp rax, r12
	jne .LBB175_21
	jmp .LBB175_23
.Ltmp19265:
.LBB175_2:
	mov qword ptr [rsp + 40], r14
	mov qword ptr [rsp + 32], r13
.Ltmp19266:
	lea r13, [r13 + 4*r13 + 5]
	xor r14d, r14d
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.188]
	mov qword ptr [rsp + 16], rax
	jmp .LBB175_3
.Ltmp19267:
	.p2align	4
.LBB175_39:
.Ltmp19238:
	mov rdi, r15
	mov rsi, rbp
	mov rdx, qword ptr [rsp + 48]
	call masstree::leaf24::LeafNode24<S>::clear_ksuf
.Ltmp19269:
.LBB175_32:
	inc r14
.Ltmp19270:
	add r13, 5
	cmp qword ptr [rsp + 8], r14
.Ltmp19271:
	je .LBB175_10
.Ltmp19272:
.LBB175_3:
	mov rax, qword ptr [rsp + 24]
.Ltmp19273:
	mov ecx, r13d
	mov rbp, qword ptr [rsp + 56]
	shrd rax, rbp, cl
	shr rbp, cl
	test r13b, 64
	cmove rbp, rax
	and ebp, 31
.Ltmp19274:
	cmp rbp, 24
	jae .LBB175_4
.Ltmp19275:
	mov rcx, qword ptr [r15 + 8*rbp + 128]
.Ltmp19276:
	movzx eax, byte ptr [r15 + rbp + 320]
.Ltmp19277:
	cmp r14, 24
	je .LBB175_30
.Ltmp19278:
	mov qword ptr [rbx + 8*r14 + 128], rcx
.Ltmp19279:
	mov byte ptr [rbx + r14 + 320], al
.Ltmp19280:
	xor ecx, ecx
.Ltmp19281:
	xchg qword ptr [r15 + 8*rbp + 344], rcx
.Ltmp19282:
	mov qword ptr [rbx + 8*r14 + 344], rcx
.Ltmp19283:
	cmp al, 64
	jne .LBB175_32
.Ltmp19284:
	movzx eax, byte ptr [r15 + rbp + 320]
.Ltmp19285:
	cmp al, 64
.Ltmp19286:
	jne .LBB175_39
.Ltmp19287:
	mov rdi, qword ptr [r15 + 536]
.Ltmp19288:
	test rdi, rdi
.Ltmp19289:
	je .LBB175_39
.Ltmp19290:
	mov edx, dword ptr [rdi + 8*rbp + 24]
	mov eax, 4294967295
	cmp rdx, rax
	je .LBB175_39
	movzx ecx, word ptr [rdi + 8*rbp + 28]
.Ltmp19293:
	lea rsi, [rcx + rdx]
.Ltmp19294:
	mov rax, qword ptr [rdi + 16]
.Ltmp19295:
	cmp rsi, rax
.Ltmp19296:
	ja .LBB175_37
.Ltmp19297:
	add rdx, qword ptr [rdi + 8]
.Ltmp19298:
	mov rdi, rbx
.Ltmp19299:
	mov rsi, r14
.Ltmp19300:
	mov r8, qword ptr [rsp + 48]
	call masstree::leaf24::LeafNode24<S>::assign_ksuf
.Ltmp19301:
.Ltmp19235:
	jmp .LBB175_39
.Ltmp19302:
.LBB175_10:
	cmp qword ptr [rsp + 8], 24
	jne .LBB175_12
	movabs rdx, 1708387328366441304
	movabs rbp, -5393897070460337128
	mov r14, qword ptr [rsp + 40]
	mov r13, qword ptr [rsp + 32]
	jmp .LBB175_23
.Ltmp19304:
.LBB175_19:
	mov r8, qword ptr [rsp + 8]
	lea ecx, [r8 + 4*r8]
	inc r8
	add cl, 5
	mov ebp, 23
	xor edx, edx
	shld rdx, rbp, cl
	shl rbp, cl
	xor r9d, r9d
	test cl, 64
	cmovne rdx, rbp
	cmovne rbp, r9
	or rdx, rsi
	or rbp, rdi
	mov rdi, rbp
	mov rsi, rdx
	mov qword ptr [rsp + 8], r8
.Ltmp19305:
	cmp rax, r12
	je .LBB175_23
.LBB175_21:
	mov rcx, qword ptr [rsp + 8]
	lea r8, [rcx - 24]
	lea rax, [rcx + 4*rcx]
	add rax, 5
	sub r12, rcx
	sub r12, r13
	add r12, 23
	xor r9d, r9d
	mov rbp, rdi
	mov rdx, rsi
	.p2align	4
.LBB175_22:
	xor esi, esi
	mov ecx, eax
	shld rsi, r12, cl
	mov rdi, r12
	shl rdi, cl
	test al, 64
	cmovne rsi, rdi
	cmovne rdi, r9
	or rsi, rdx
	or rdi, rbp
	lea rbp, [r12 - 1]
	lea ecx, [rax + 5]
	xor edx, edx
	shld rdx, rbp, cl
	shl rbp, cl
	test cl, 64
	cmovne rdx, rbp
	cmovne rbp, r9
	or rdx, rsi
	or rbp, rdi
	add rax, 10
	add r12, -2
	add r8, 2
	jne .LBB175_22
	jmp .LBB175_23
.Ltmp19309:
.LBB175_12:
	mov r13, qword ptr [rsp + 32]
	lea rax, [r13 + 1]
	cmp r12, rax
	jne .LBB175_27
	xor esi, esi
	mov rdi, qword ptr [rsp + 8]
	xor edx, edx
	jmp .LBB175_14
.LBB175_27:
	xor r8d, r8d
	mov r14, qword ptr [rsp + 8]
	mov r9, r14
	and r9, -2
	mov eax, 5
	xor esi, esi
	xor edx, edx
	.p2align	4
.LBB175_28:
	xor r10d, r10d
	mov ecx, eax
	shld r10, rdx, cl
	mov r11, rdx
	shl r11, cl
	test al, 64
	cmovne r10, r11
	cmovne r11, r8
	lea rdi, [rdx + 1]
	or r10, rsi
	or r11, r14
	lea ebp, [rax + 5]
	mov ecx, ebp
	and cl, 62
	xor esi, esi
	shld rsi, rdi, cl
	shl rdi, cl
	add rdx, 2
	test bpl, 64
	cmovne rsi, rdi
	cmovne rdi, r8
	or rsi, r10
	or rdi, r11
	add rax, 10
	mov r14, rdi
	cmp r9, rdx
	jne .LBB175_28
.LBB175_14:
	test byte ptr [rsp + 8], 1
	je .LBB175_16
	lea ecx, [rdx + 4*rdx]
	add cl, 5
	xor eax, eax
	shld rax, rdx, cl
	shl rdx, cl
	xor r8d, r8d
	test cl, 64
	cmovne rax, rdx
	cmovne rdx, r8
	or rsi, rax
	or rdi, rdx
.LBB175_16:
	mov rbp, rdi
	mov rdx, rsi
	cmp qword ptr [rsp + 8], 23
	mov r14, qword ptr [rsp + 40]
	jbe .LBB175_17
.Ltmp19317:
.LBB175_23:
	lea rdi, [rbx + 64]
.Ltmp19318:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC]
.Ltmp19319:
.Ltmp19241:
	mov rsi, rbp
	call rax
.Ltmp19320:
	mov rsi, qword ptr [rsp + 24]
.Ltmp19321:
	and rsi, -32
.Ltmp19322:
	or rsi, r13
.Ltmp19323:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC]
	mov rdi, qword ptr [rsp + 64]
	mov rdx, qword ptr [rsp + 56]
.Ltmp19325:
	call rax
.Ltmp19326:
	shr ebp, 5
.Ltmp19327:
	and ebp, 31
.Ltmp19328:
	cmp ebp, 23
	ja .LBB175_4
.Ltmp19329:
	mov rax, qword ptr [rbx + 8*rbp + 128]
.Ltmp19330:
	mov qword ptr [r14], rbx
	mov qword ptr [r14 + 8], rax
	mov byte ptr [r14 + 16], 0
.Ltmp19331:
	add rsp, 72
	.cfi_def_cfa_offset 56
	pop rbx
.Ltmp19332:
	.cfi_def_cfa_offset 48
	pop r12
	.cfi_def_cfa_offset 40
	pop r13
.Ltmp19333:
	.cfi_def_cfa_offset 32
	pop r14
	.cfi_def_cfa_offset 24
	pop r15
.Ltmp19334:
	.cfi_def_cfa_offset 16
	pop rbp
	.cfi_def_cfa_offset 8
	ret
.Ltmp19336:
.LBB175_37:
	.cfi_def_cfa_offset 128
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.197]
	mov rdi, rdx
.Ltmp19337:
	mov rdx, rax
.Ltmp19338:
	call core::slice::index::slice_index_fail
.Ltmp19339:
	jmp .LBB175_6
.Ltmp19340:
.LBB175_4:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
	mov qword ptr [rsp + 16], rax
	jmp .LBB175_5
.Ltmp19341:
.LBB175_30:
	mov ebp, 24
.Ltmp19342:
.LBB175_5:
	mov esi, 24
	mov rdi, rbp
	mov rdx, qword ptr [rsp + 16]
	call core::panicking::panic_bounds_check
.Ltmp19343:
.Ltmp19246:
.LBB175_6:
	ud2
.Ltmp19344:
.Ltmp19240:
	jmp .LBB175_9
.Ltmp19345:
.Ltmp19247:
.LBB175_9:
	mov r14, rax
	mov rdi, rbx
	call core::ptr::drop_in_place<alloc::boxed::Box<masstree::leaf24::LeafNode24<masstree::value::LeafValue<u64>>>>
	mov rdi, r14
	call _Unwind_Resume@PLT
.Lfunc_end175:
	.size	<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated, .Lfunc_end175-<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated
	.cfi_endproc
.section ".gcc_except_table.<masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated","a",@progbits
	.p2align	2, 0x0
GCC_except_table175:
<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated:
.Lfunc_begin216:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception88
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
	sub rsp, 408
	.cfi_def_cfa_offset 464
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	.cfi_offset rbp, -16
	mov r9, rcx
	mov eax, dword ptr [rsi]
	and eax, -2147483648
	or eax, 5
.Ltmp23544:
	mov dword ptr [rcx], eax
.Ltmp23545:
	mov rax, qword ptr [rsi + 56]
.Ltmp23546:
	mov qword ptr [rsp + 88], rax
.Ltmp23547:
	mov ebx, eax
	and ebx, 15
.Ltmp23548:
	mov r11, rbx
	sub r11, rdx
.Ltmp23549:
	je .LBB216_13
	mov qword ptr [rsp + 368], rbx
.Ltmp23551:
	mov qword ptr [rsp + 24], rsi
.Ltmp23552:
	mov qword ptr [rsp + 384], rdi
	mov qword ptr [rsp], r9
	mov r15, qword ptr [r8]
	mov rax, qword ptr [r8 + 16]
	mov qword ptr [rsp + 80], rax
	mov rax, qword ptr [r8 + 24]
	mov qword ptr [rsp + 72], rax
	lea rax, [r15 + 8*rax]
	mov qword ptr [rsp + 40], rax
	mov qword ptr [rsp + 376], rdx
.Ltmp23553:
	lea rcx, [4*rdx + 4]
.Ltmp23554:
	xor ebp, ebp
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.183]
	mov qword ptr [rsp + 56], rax
	mov qword ptr [rsp + 392], r11
.Ltmp23555:
	mov qword ptr [rsp + 48], r15
	jmp .LBB216_2
.Ltmp23556:
.LBB216_99:
	mov byte ptr [rsi + r12 + 184], 0
.Ltmp23557:
.LBB216_29:
	inc rbp
.Ltmp23558:
	mov rcx, qword ptr [rsp + 400]
.Ltmp23559:
	add rcx, 4
	mov r11, qword ptr [rsp + 392]
	cmp r11, rbp
.Ltmp23560:
	je .LBB216_11
.Ltmp23561:
.LBB216_2:
	mov qword ptr [rsp + 400], rcx
.Ltmp23562:
	and cl, 60
	mov r12, qword ptr [rsp + 88]
	shr r12, cl
	and r12d, 15
.Ltmp23563:
	cmp r12, 15
	je .LBB216_25
	mov rsi, qword ptr [rsp + 24]
.Ltmp23565:
	mov rcx, qword ptr [rsi + 8*r12 + 64]
.Ltmp23566:
	movzx eax, byte ptr [rsi + r12 + 184]
.Ltmp23567:
	cmp rbp, 15
	je .LBB216_4
	mov rdx, qword ptr [rsp]
.Ltmp23569:
	mov qword ptr [rdx + 8*rbp + 64], rcx
.Ltmp23570:
	mov byte ptr [rdx + rbp + 184], al
.Ltmp23571:
	xor ecx, ecx
.Ltmp23572:
	xchg qword ptr [rsi + 8*r12 + 200], rcx
.Ltmp23573:
	mov qword ptr [rdx + 8*rbp + 200], rcx
.Ltmp23574:
	cmp al, 64
	jne .LBB216_29
.Ltmp23575:
	movzx eax, byte ptr [rsi + r12 + 184]
.Ltmp23576:
	cmp al, 64
.Ltmp23577:
	jne .LBB216_87
.Ltmp23578:
	mov rax, qword ptr [rsi + 320]
.Ltmp23579:
	test rax, rax
.Ltmp23580:
	je .LBB216_87
.Ltmp23581:
	mov r13d, dword ptr [rax + 8*r12 + 24]
	mov ecx, 4294967295
.Ltmp23582:
	cmp r13, rcx
.Ltmp23583:
	je .LBB216_87
.Ltmp23584:
	movzx ebx, word ptr [rax + 8*r12 + 28]
.Ltmp23585:
	movzx ecx, bx
	lea rsi, [rcx + r13]
.Ltmp23586:
	mov rdx, qword ptr [rax + 16]
.Ltmp23587:
	cmp rsi, rdx
.Ltmp23588:
	ja .LBB216_34
.Ltmp23589:
	add r13, qword ptr [rax + 8]
.Ltmp23590:
	mov rax, qword ptr [rsp]
.Ltmp23591:
	mov r15, qword ptr [rax + 320]
.Ltmp23592:
	test r15, r15
	mov qword ptr [rsp + 16], r15
.Ltmp23593:
	je .LBB216_49
.Ltmp23594:
	mov edi, dword ptr [r15 + 8*rbp + 24]
	mov eax, 4294967295
	cmp rdi, rax
	je .LBB216_38
	cmp bx, word ptr [r15 + 8*rbp + 28]
	ja .LBB216_38
.Ltmp23597:
	lea rsi, [rcx + rdi]
	mov rdx, qword ptr [r15 + 16]
.Ltmp23598:
	cmp rsi, rdx
.Ltmp23599:
	ja .LBB216_41
.Ltmp23600:
	add rdi, qword ptr [r15 + 8]
.Ltmp23601:
	mov rsi, r13
.Ltmp23602:
	mov rdx, rcx
.Ltmp23603:
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp23604:
	mov word ptr [r15 + 8*rbp + 28], bx
	jmp .LBB216_86
.Ltmp23606:
.LBB216_49:
	mov qword ptr [rsp + 32], rcx
.Ltmp23607:
	mov word ptr [rsp + 8], bx
	mov r14d, 128
	mov r15d, 1
.Ltmp23608:
	mov edi, 128
	mov esi, 1
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23609:
	test rax, rax
	je .LBB216_48
.Ltmp23610:
	mov dword ptr [rsp + 120], -1
	mov word ptr [rsp + 124], 0
	mov dword ptr [rsp + 128], -1
	mov word ptr [rsp + 132], 0
	mov dword ptr [rsp + 136], -1
	mov word ptr [rsp + 140], 0
	mov dword ptr [rsp + 144], -1
	mov word ptr [rsp + 148], 0
	mov dword ptr [rsp + 152], -1
	mov word ptr [rsp + 156], 0
	mov dword ptr [rsp + 160], -1
	mov word ptr [rsp + 164], 0
	mov dword ptr [rsp + 168], -1
	mov word ptr [rsp + 172], 0
	mov dword ptr [rsp + 176], -1
	mov word ptr [rsp + 180], 0
	mov dword ptr [rsp + 184], -1
	mov word ptr [rsp + 188], 0
	mov dword ptr [rsp + 192], -1
	mov word ptr [rsp + 196], 0
	mov dword ptr [rsp + 200], -1
	mov word ptr [rsp + 204], 0
	mov dword ptr [rsp + 208], -1
	mov word ptr [rsp + 212], 0
	mov dword ptr [rsp + 216], -1
	mov word ptr [rsp + 220], 0
	mov dword ptr [rsp + 224], -1
	mov word ptr [rsp + 228], 0
	mov dword ptr [rsp + 232], -1
	mov word ptr [rsp + 236], 0
	mov qword ptr [rsp + 96], 128
	mov qword ptr [rsp + 104], rax
	mov qword ptr [rsp + 112], 0
	mov r14d, 128
	xor ebx, ebx
	jmp .LBB216_51
.Ltmp23611:
.LBB216_38:
	mov word ptr [rsp + 8], bx
.Ltmp23612:
	mov r14, qword ptr [r15 + 16]
.Ltmp23613:
	lea rbx, [r14 + rcx]
	cmp rbx, qword ptr [r15]
	jbe .LBB216_39
.Ltmp23614:
	mov qword ptr [rsp + 32], rcx
.Ltmp23615:
	lea rax, [r15 + 24]
.Ltmp23616:
	mov rcx, qword ptr [rax + 112]
	mov qword ptr [rsp + 352], rcx
	movups xmm0, xmmword ptr [rax + 96]
	movaps xmmword ptr [rsp + 336], xmm0
	movups xmm0, xmmword ptr [rax + 80]
	movaps xmmword ptr [rsp + 320], xmm0
	movups xmm0, xmmword ptr [rax + 64]
	movaps xmmword ptr [rsp + 304], xmm0
	movdqu xmm0, xmmword ptr [rax]
	movdqu xmm1, xmmword ptr [rax + 16]
	movdqu xmm2, xmmword ptr [rax + 32]
	movdqu xmm3, xmmword ptr [rax + 48]
	movdqa xmmword ptr [rsp + 288], xmm3
	movdqa xmmword ptr [rsp + 272], xmm2
	movdqa xmmword ptr [rsp + 256], xmm1
	movdqa xmmword ptr [rsp + 240], xmm0
	mov rbx, qword ptr [r15 + 8]
.Ltmp23617:
	test r14, r14
	je .LBB216_45
.Ltmp23618:
	mov r15d, 1
.Ltmp23619:
	mov esi, 1
	mov rdi, r14
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23620:
	test rax, rax
	jne .LBB216_46
	jmp .LBB216_48
.Ltmp23621:
.LBB216_39:
	mov rdi, qword ptr [r15 + 8]
.Ltmp23622:
	add rdi, r14
.Ltmp23623:
	mov rsi, r13
	mov rdx, rcx
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp23624:
	mov qword ptr [r15 + 16], rbx
.Ltmp23625:
	mov dword ptr [r15 + 8*rbp + 24], r14d
	movzx ebx, word ptr [rsp + 8]
.Ltmp23626:
	mov word ptr [r15 + 8*rbp + 28], bx
	jmp .LBB216_86
.Ltmp23628:
.LBB216_45:
	mov eax, 1
.Ltmp23629:
.LBB216_46:
	mov rdi, rax
	mov rsi, rbx
	mov rdx, r14
	mov rbx, rax
.Ltmp23630:
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp23631:
	movaps xmm0, xmmword ptr [rsp + 240]
	movaps xmm1, xmmword ptr [rsp + 256]
	movaps xmm2, xmmword ptr [rsp + 272]
	movaps xmm3, xmmword ptr [rsp + 288]
	lea rcx, [rsp + 96]
	movups xmmword ptr [rcx + 24], xmm0
	movups xmmword ptr [rcx + 40], xmm1
	movups xmmword ptr [rcx + 56], xmm2
	movups xmmword ptr [rcx + 72], xmm3
	movaps xmm0, xmmword ptr [rsp + 304]
	movups xmmword ptr [rcx + 88], xmm0
	movaps xmm0, xmmword ptr [rsp + 320]
	movups xmmword ptr [rcx + 104], xmm0
	movaps xmm0, xmmword ptr [rsp + 336]
	movups xmmword ptr [rcx + 120], xmm0
	mov rax, qword ptr [rsp + 352]
	mov qword ptr [rcx + 136], rax
	mov rax, rbx
	mov qword ptr [rsp + 96], r14
	mov qword ptr [rsp + 104], rbx
	mov qword ptr [rsp + 112], r14
	mov rbx, r14
.Ltmp23632:
.LBB216_51:
	mov rdx, qword ptr [rsp + 32]
.Ltmp23633:
	sub r14, rbx
.Ltmp23634:
	mov qword ptr [rsp + 64], rbx
.Ltmp23635:
	cmp r14, rdx
	mov r15, qword ptr [rsp + 48]
.Ltmp23636:
	jb .LBB216_52
.Ltmp23637:
.LBB216_54:
	mov r14, rax
.Ltmp23638:
	lea rdi, [rax + rbx]
.Ltmp23639:
	mov rsi, r13
	mov r13, rdx
.Ltmp23640:
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp23641:
	add rbx, r13
.Ltmp23642:
	mov qword ptr [rsp + 112], rbx
.Ltmp23643:
	cmp rbp, 15
	jae .LBB216_55
.Ltmp23644:
	mov rax, qword ptr [rsp + 64]
	mov dword ptr [rsp + 8*rbp + 120], eax
	movzx eax, word ptr [rsp + 8]
	mov word ptr [rsp + 8*rbp + 124], ax
.Ltmp23645:
	mov rbx, qword ptr [rsp + 96]
.Ltmp23646:
	mov edi, 144
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23647:
	test rax, rax
	je .LBB216_57
.Ltmp23648:
	mov qword ptr [rax], rbx
	mov qword ptr [rax + 8], r14
	lea rcx, [rsp + 96]
	movups xmm0, xmmword ptr [rcx + 16]
	movdqu xmm1, xmmword ptr [rcx + 32]
	movdqu xmm2, xmmword ptr [rcx + 48]
	movdqu xmm3, xmmword ptr [rcx + 64]
	movups xmmword ptr [rax + 16], xmm0
	movdqu xmmword ptr [rax + 32], xmm1
	movdqu xmmword ptr [rax + 48], xmm2
	movdqu xmmword ptr [rax + 64], xmm3
	movups xmm0, xmmword ptr [rcx + 80]
	movups xmmword ptr [rax + 80], xmm0
	movups xmm0, xmmword ptr [rcx + 96]
	movups xmmword ptr [rax + 96], xmm0
	movups xmm0, xmmword ptr [rcx + 112]
	movups xmmword ptr [rax + 112], xmm0
	movdqu xmm0, xmmword ptr [rcx + 128]
	movdqu xmmword ptr [rax + 128], xmm0
	mov rcx, qword ptr [rsp]
.Ltmp23650:
	mov qword ptr [rcx + 320], rax
.Ltmp23651:
	cmp qword ptr [rsp + 16], 0
.Ltmp23652:
	je .LBB216_86
.Ltmp23653:
	mov rax, qword ptr [rsp + 40]
.Ltmp23654:
	mov rax, qword ptr [rax]
.Ltmp23655:
	test rax, rax
	je .LBB216_63
.LBB216_64:
	mov rcx, qword ptr [rsp + 80]
.Ltmp23658:
	shl rcx, 8
	lea r13, [rax + rcx]
.Ltmp23659:
	movzx eax, byte ptr [rax + rcx + 128]
.Ltmp23660:
	test al, al
	je .LBB216_65
.Ltmp23661:
	mov r14, qword ptr [r13]
	test r14, r14
	jne .LBB216_78
.Ltmp23662:
.LBB216_67:
	mov rbx, qword ptr [r15 + 952]
.Ltmp23663:
	mov r14, rbx
	shl r14, 5
	mov rax, rbx
	shr rax, 59
	sete al
.Ltmp23664:
	movabs rcx, 9223372036854775800
	cmp r14, rcx
	setbe cl
.Ltmp23665:
	test al, cl
	je .LBB216_68
.Ltmp23666:
	test r14, r14
	je .LBB216_70
.Ltmp23667:
	mov r15d, 8
.Ltmp23668:
	mov esi, 8
	mov rdi, r14
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	mov qword ptr [rsp + 8], rax
.Ltmp23670:
	test rax, rax
	jne .LBB216_72
	jmp .LBB216_48
.Ltmp23671:
.LBB216_70:
	mov eax, 8
	mov qword ptr [rsp + 8], rax
	xor ebx, ebx
.Ltmp23672:
.LBB216_72:
	mov edi, 32
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23673:
	test rax, rax
	je .LBB216_73
.Ltmp23674:
	mov r14, rax
	mov qword ptr [rax], rbx
	mov rax, qword ptr [rsp + 8]
	mov qword ptr [r14 + 8], rax
	pxor xmm0, xmm0
	movdqu xmmword ptr [r14 + 16], xmm0
.Ltmp23675:
	mov qword ptr [r13], r14
	mov r15, qword ptr [rsp + 48]
.Ltmp23676:
.LBB216_78:
	cmp r14, -1
	je .LBB216_83
.Ltmp23677:
	mov rbx, qword ptr [r14 + 16]
.Ltmp23678:
	cmp rbx, qword ptr [r14]
	jne .LBB216_81
.Ltmp23510:
	mov rdi, r14
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp23681:
.LBB216_81:
	mov rax, qword ptr [r14 + 8]
.Ltmp23682:
	mov rcx, rbx
	shl rcx, 5
.Ltmp23683:
	lea rdx, [rip + core::ops::function::FnOnce::call_once]
	mov qword ptr [rax + rcx], rdx
	mov rdx, qword ptr [rsp + 16]
	mov qword ptr [rax + rcx + 8], rdx
	mov qword ptr [rax + rcx + 16], 0
	mov qword ptr [rax + rcx + 24], r14
.Ltmp23684:
	inc rbx
.Ltmp23685:
	mov qword ptr [r14 + 16], rbx
.Ltmp23686:
	cmp rbx, qword ptr [r15 + 952]
	jb .LBB216_86
.Ltmp23687:
	mov rdi, r15
	mov rsi, r13
	call seize::raw::collector::Collector::try_retire
.Ltmp23513:
	jmp .LBB216_86
.Ltmp23688:
.LBB216_83:
	mov r14, qword ptr [rsp + 16]
.Ltmp23689:
	cmp qword ptr [r14], 0
	mov rbx, qword ptr [rip + mi_free@GOTPCREL]
	je .LBB216_85
.Ltmp23690:
	mov rdi, qword ptr [r14 + 8]
.Ltmp23691:
	call rbx
.Ltmp23692:
.LBB216_85:
	mov rdi, r14
	call rbx
.Ltmp23693:
.LBB216_86:
	mov rax, qword ptr [rsp]
.Ltmp23694:
	mov byte ptr [rax + rbp + 184], 64
	mov rsi, qword ptr [rsp + 24]
.Ltmp23695:
	.p2align	4
.LBB216_87:
	mov r13, qword ptr [rsi + 320]
.Ltmp23697:
	test r13, r13
.Ltmp23698:
	je .LBB216_99
.Ltmp23699:
	mov rax, qword ptr [r13 + 136]
	mov qword ptr [rsp + 352], rax
	movups xmm0, xmmword ptr [r13 + 120]
	movaps xmmword ptr [rsp + 336], xmm0
	movups xmm0, xmmword ptr [r13 + 104]
	movaps xmmword ptr [rsp + 320], xmm0
	movups xmm0, xmmword ptr [r13 + 88]
	movaps xmmword ptr [rsp + 304], xmm0
	movdqu xmm0, xmmword ptr [r13 + 24]
	movdqu xmm1, xmmword ptr [r13 + 40]
	movdqu xmm2, xmmword ptr [r13 + 56]
	movdqu xmm3, xmmword ptr [r13 + 72]
	movdqa xmmword ptr [rsp + 288], xmm3
	movdqa xmmword ptr [rsp + 272], xmm2
	movdqa xmmword ptr [rsp + 256], xmm1
	movdqa xmmword ptr [rsp + 240], xmm0
	mov r14, qword ptr [r13 + 16]
.Ltmp23700:
	test r14, r14
.Ltmp23701:
	js .LBB216_68
.Ltmp23702:
	mov rsi, qword ptr [r13 + 8]
.Ltmp23703:
	je .LBB216_90
.Ltmp23704:
	mov rbx, rsi
.Ltmp23705:
	mov esi, 1
	mov rdi, r14
	mov rax, qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	call rax
	mov rsi, rbx
	mov rbx, rax
.Ltmp23706:
	test rax, rax
	jne .LBB216_91
	jmp .LBB216_98
.Ltmp23707:
.LBB216_90:
	mov ebx, 1
.Ltmp23708:
.LBB216_91:
	mov rdi, rbx
	mov rdx, r14
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp23709:
	movaps xmm0, xmmword ptr [rsp + 240]
	movaps xmm1, xmmword ptr [rsp + 256]
	movaps xmm2, xmmword ptr [rsp + 272]
	movaps xmm3, xmmword ptr [rsp + 288]
	lea r15, [rsp + 96]
	movups xmmword ptr [r15 + 24], xmm0
	movups xmmword ptr [r15 + 40], xmm1
	movups xmmword ptr [r15 + 56], xmm2
	movups xmmword ptr [r15 + 72], xmm3
	movaps xmm0, xmmword ptr [rsp + 304]
	movups xmmword ptr [r15 + 88], xmm0
	movaps xmm0, xmmword ptr [rsp + 320]
	movups xmmword ptr [r15 + 104], xmm0
	movaps xmm0, xmmword ptr [rsp + 336]
	movups xmmword ptr [r15 + 120], xmm0
	mov rax, qword ptr [rsp + 352]
	mov qword ptr [r15 + 136], rax
	mov qword ptr [rsp + 96], r14
	mov qword ptr [rsp + 104], rbx
	mov qword ptr [rsp + 112], r14
.Ltmp23710:
	mov dword ptr [rsp + 8*r12 + 120], -1
	mov word ptr [rsp + 8*r12 + 124], 0
.Ltmp23711:
	mov edi, 144
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23712:
	test rax, rax
	je .LBB216_92
.Ltmp23713:
	mov qword ptr [rax], r14
	mov qword ptr [rax + 8], rbx
	movups xmm0, xmmword ptr [r15 + 16]
	movdqu xmm1, xmmword ptr [r15 + 32]
	movdqu xmm2, xmmword ptr [r15 + 48]
	movdqu xmm3, xmmword ptr [r15 + 64]
	movups xmmword ptr [rax + 16], xmm0
	movdqu xmmword ptr [rax + 32], xmm1
	movdqu xmmword ptr [rax + 48], xmm2
	movdqu xmmword ptr [rax + 64], xmm3
	movups xmm0, xmmword ptr [r15 + 80]
	movups xmmword ptr [rax + 80], xmm0
	movups xmm0, xmmword ptr [r15 + 96]
	movups xmmword ptr [rax + 96], xmm0
	movups xmm0, xmmword ptr [r15 + 112]
	movups xmmword ptr [rax + 112], xmm0
	movdqu xmm0, xmmword ptr [r15 + 128]
	movdqu xmmword ptr [rax + 128], xmm0
	mov rcx, qword ptr [rsp + 24]
.Ltmp23715:
	mov qword ptr [rcx + 320], rax
.Ltmp23716:
	mov rax, qword ptr [rsp + 40]
.Ltmp23717:
	mov rax, qword ptr [rax]
.Ltmp23718:
	test rax, rax
	mov r15, qword ptr [rsp + 48]
	je .LBB216_103
.Ltmp23720:
.LBB216_104:
	mov rcx, qword ptr [rsp + 80]
.Ltmp23721:
	shl rcx, 8
	lea rbx, [rax + rcx]
.Ltmp23722:
	movzx eax, byte ptr [rax + rcx + 128]
.Ltmp23723:
	test al, al
	je .LBB216_105
.Ltmp23724:
	mov qword ptr [rsp + 16], rbx
.Ltmp23725:
	mov r14, qword ptr [rbx]
	test r14, r14
	je .LBB216_107
.Ltmp23726:
.LBB216_115:
	cmp r14, -1
	je .LBB216_120
.Ltmp23727:
.LBB216_116:
	mov rbx, qword ptr [r14 + 16]
.Ltmp23728:
	cmp rbx, qword ptr [r14]
	jne .LBB216_118
.Ltmp23526:
	mov rdi, r14
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp23731:
.LBB216_118:
	mov rax, qword ptr [r14 + 8]
.Ltmp23732:
	mov rcx, rbx
	shl rcx, 5
.Ltmp23733:
	lea rdx, [rip + core::ops::function::FnOnce::call_once]
	mov qword ptr [rax + rcx], rdx
	mov qword ptr [rax + rcx + 8], r13
	mov qword ptr [rax + rcx + 16], 0
	mov qword ptr [rax + rcx + 24], r14
.Ltmp23734:
	inc rbx
.Ltmp23735:
	mov qword ptr [r14 + 16], rbx
.Ltmp23736:
	cmp rbx, qword ptr [r15 + 952]
	jb .LBB216_123
.Ltmp23737:
	mov rdi, r15
	mov rsi, qword ptr [rsp + 16]
	call seize::raw::collector::Collector::try_retire
	jmp .LBB216_123
.Ltmp23738:
.LBB216_103:
	mov rdi, qword ptr [rsp + 40]
	mov rsi, qword ptr [rsp + 72]
.Ltmp23739:
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp23740:
.Ltmp23523:
	jmp .LBB216_104
.Ltmp23741:
.LBB216_105:
	mov rdi, rbx
	call seize::raw::tls::ThreadLocal<T>::write
	mov qword ptr [rsp + 16], rbx
.Ltmp23743:
	mov r14, qword ptr [rbx]
	test r14, r14
	jne .LBB216_115
.Ltmp23744:
.LBB216_107:
	mov rbx, qword ptr [r15 + 952]
.Ltmp23745:
	mov r14, rbx
	shl r14, 5
	mov rax, rbx
	shr rax, 59
	setne al
.Ltmp23746:
	movabs rcx, 9223372036854775800
	cmp r14, rcx
	seta cl
.Ltmp23747:
	or cl, al
	jne .LBB216_68
.Ltmp23748:
	test r14, r14
	je .LBB216_109
.Ltmp23749:
	mov r15d, 8
.Ltmp23750:
	mov esi, 8
	mov rdi, r14
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	mov qword ptr [rsp + 8], rax
.Ltmp23752:
	test rax, rax
	jne .LBB216_111
	jmp .LBB216_48
.Ltmp23753:
.LBB216_109:
	mov eax, 8
	mov qword ptr [rsp + 8], rax
	xor ebx, ebx
.Ltmp23754:
.LBB216_111:
	mov edi, 32
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23755:
	test rax, rax
	je .LBB216_112
.Ltmp23756:
	mov r14, rax
	mov qword ptr [rax], rbx
	mov rax, qword ptr [rsp + 8]
	mov qword ptr [r14 + 8], rax
	pxor xmm0, xmm0
	movdqu xmmword ptr [r14 + 16], xmm0
	mov rax, qword ptr [rsp + 16]
.Ltmp23757:
	mov qword ptr [rax], r14
	mov r15, qword ptr [rsp + 48]
.Ltmp23758:
	cmp r14, -1
	jne .LBB216_116
.Ltmp23759:
.LBB216_120:
	cmp qword ptr [r13], 0
	mov rbx, qword ptr [rip + mi_free@GOTPCREL]
	je .LBB216_122
.Ltmp23760:
	mov rdi, qword ptr [r13 + 8]
.Ltmp23761:
	call rbx
.Ltmp23762:
.LBB216_122:
	mov rdi, r13
	call rbx
.Ltmp23763:
.LBB216_123:
	mov rax, qword ptr [rsp + 24]
.Ltmp23764:
	mov byte ptr [rax + r12 + 184], 0
.Ltmp23765:
	jmp .LBB216_29
.Ltmp23766:
.LBB216_52:
.Ltmp23502:
	mov ecx, 1
	mov r8d, 1
	lea rdi, [rsp + 96]
	mov rsi, qword ptr [rsp + 64]
	mov qword ptr [rsp + 32], rdx
.Ltmp23767:
	call alloc::raw_vec::RawVecInner<A>::reserve::do_reserve_and_handle
.Ltmp23768:
.Ltmp23503:
	mov rax, qword ptr [rsp + 104]
.Ltmp23769:
	mov rbx, qword ptr [rsp + 112]
	mov rdx, qword ptr [rsp + 32]
	jmp .LBB216_54
.Ltmp23770:
.LBB216_63:
.Ltmp23508:
	mov rdi, qword ptr [rsp + 40]
	mov rsi, qword ptr [rsp + 72]
.Ltmp23771:
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp23772:
.Ltmp23509:
	jmp .LBB216_64
.Ltmp23773:
.LBB216_65:
	mov rdi, r13
	call seize::raw::tls::ThreadLocal<T>::write
.Ltmp23774:
	mov r14, qword ptr [r13]
	test r14, r14
	jne .LBB216_78
	jmp .LBB216_67
.Ltmp23775:
.LBB216_68:
.Ltmp23537:
	call alloc::raw_vec::capacity_overflow
.Ltmp23538:
	jmp .LBB216_26
.Ltmp23776:
.LBB216_11:
	cmp r11, 15
	mov r9, qword ptr [rsp]
	mov rdi, qword ptr [rsp + 384]
	mov rsi, qword ptr [rsp + 24]
	mov rdx, qword ptr [rsp + 376]
	mov rbx, qword ptr [rsp + 368]
	jne .LBB216_13
	movabs rax, -1311768467463790336
	or rax, 15
	jmp .LBB216_22
.Ltmp23778:
.LBB216_13:
	lea ecx, [4*r11]
	mov r8, -1
	shl r8, cl
	not r8
	shl r8, 4
.Ltmp23779:
	movabs rax, -1311768467463790336
.Ltmp23780:
	and rax, r8
	or rax, r11
.Ltmp23781:
	cmp r11, 14
	ja .LBB216_22
	mov r10, rdx
	sub r10, rbx
	add r10, 15
	cmp r10, 4
	jae .LBB216_16
	mov rcx, r11
	jmp .LBB216_20
.LBB216_16:
	lea r8, [r11 + 14]
.Ltmp23785:
	mov rcx, r10
	and rcx, -4
	movq xmm0, rax
	movq xmm1, r8
	pshufd xmm2, xmm1, 68
	movq xmm1, r11
	pshufd xmm3, xmm1, 68
	paddq xmm3, xmmword ptr [rip + .LCPI216_0]
	pxor xmm1, xmm1
	movdqa xmm4, xmmword ptr [rip + .LCPI216_1]
	movdqa xmm5, xmmword ptr [rip + .LCPI216_2]
	mov rax, rcx
.Ltmp23786:
	.p2align	4
.LBB216_17:
	movdqa xmm6, xmm3
	paddq xmm6, xmm4
	movdqa xmm7, xmm2
	psubq xmm7, xmm3
	movdqa xmm8, xmm2
	psubq xmm8, xmm6
	movdqa xmm9, xmm3
	psllq xmm9, 2
	psllq xmm6, 2
	paddq xmm9, xmm5
	movdqa xmm10, xmm7
	psllq xmm10, xmm9
	paddq xmm6, xmm5
	pshufd xmm9, xmm9, 238
	psllq xmm7, xmm9
	movsd xmm7, xmm10
	movdqa xmm9, xmm8
	psllq xmm9, xmm6
	por xmm0, xmm7
	pshufd xmm6, xmm6, 238
	psllq xmm8, xmm6
	movsd xmm8, xmm9
	por xmm1, xmm8
	paddq xmm3, xmm5
	add rax, -4
	jne .LBB216_17
	por xmm1, xmm0
	pshufd xmm0, xmm1, 238
	por xmm0, xmm1
	movq rax, xmm0
	cmp r10, rcx
	je .LBB216_22
	add rcx, r11
.LBB216_20:
	mov r10d, 14
	sub r10, rcx
	lea rcx, [4*rcx + 4]
.Ltmp23792:
	.p2align	4
.LBB216_21:
	lea r8, [r11 + r10]
	shl r8, cl
	or rax, r8
	add rcx, 4
	add r10, -1
	jb .LBB216_21
.Ltmp23795:
.LBB216_22:
	mov qword ptr [r9 + 56], rax
.Ltmp23796:
	mov rcx, qword ptr [rsp + 88]
.Ltmp23797:
	and rcx, -16
	or rcx, rdx
.Ltmp23798:
	mov qword ptr [rsi + 56], rcx
.Ltmp23799:
	shr eax, 4
.Ltmp23800:
	and eax, 15
.Ltmp23801:
	cmp eax, 15
	je .LBB216_23
.Ltmp23802:
	mov rax, qword ptr [r9 + 8*rax + 64]
.Ltmp23803:
	mov qword ptr [rdi], r9
	mov qword ptr [rdi + 8], rax
	mov byte ptr [rdi + 16], 0
.Ltmp23804:
	add rsp, 408
	.cfi_def_cfa_offset 56
	pop rbx
.Ltmp23805:
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
	ret
.Ltmp23806:
.LBB216_34:
	.cfi_def_cfa_offset 464
.Ltmp23520:
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.197]
	mov rdi, r13
	call core::slice::index::slice_index_fail
.Ltmp23807:
.Ltmp23521:
	jmp .LBB216_26
.Ltmp23808:
.LBB216_92:
.Ltmp23534:
	mov edi, 8
	mov esi, 144
	call alloc::alloc::handle_alloc_error
.Ltmp23535:
	jmp .LBB216_26
.Ltmp23809:
.LBB216_41:
.Ltmp23500:
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.196]
.Ltmp23810:
	call core::slice::index::slice_index_fail
.Ltmp23811:
.Ltmp23501:
	jmp .LBB216_26
.Ltmp23812:
.LBB216_57:
.Ltmp23517:
	mov edi, 8
	mov esi, 144
	call alloc::alloc::handle_alloc_error
.Ltmp23518:
	jmp .LBB216_26
.Ltmp23813:
.LBB216_112:
.Ltmp23531:
	mov edi, 8
	mov esi, 32
	call alloc::alloc::handle_alloc_error
.Ltmp23532:
	jmp .LBB216_26
.Ltmp23814:
.LBB216_73:
.Ltmp23514:
	mov edi, 8
	mov esi, 32
	call alloc::alloc::handle_alloc_error
.Ltmp23515:
	jmp .LBB216_26
.Ltmp23815:
.LBB216_4:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.184]
.Ltmp23816:
.LBB216_24:
	mov qword ptr [rsp + 56], rax
.Ltmp23817:
.LBB216_25:
.Ltmp23540:
	mov edi, 15
	mov esi, 15
	mov rdx, qword ptr [rsp + 56]
	call core::panicking::panic_bounds_check
.Ltmp23818:
.Ltmp23541:
	jmp .LBB216_26
.Ltmp23819:
.LBB216_23:
	mov qword ptr [rsp], r9
.Ltmp23820:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.183]
	jmp .LBB216_24
.Ltmp23821:
.LBB216_98:
	mov r15d, 1
.Ltmp23822:
.LBB216_48:
.Ltmp23524:
	mov rdi, r15
	mov rsi, r14
	call alloc::raw_vec::handle_error
.Ltmp23525:
	jmp .LBB216_26
.Ltmp23823:
.LBB216_55:
.Ltmp23505:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.200]
	mov esi, 15
	mov rdi, rbp
	call core::panicking::panic_bounds_check
.Ltmp23824:
.Ltmp23506:
.LBB216_26:
	ud2
.Ltmp23825:
.Ltmp23504:
	jmp .LBB216_95
.Ltmp23826:
.Ltmp23507:
.LBB216_95:
	mov r15, rax
.Ltmp23827:
	lea rax, [rsp + 96]
	cmp qword ptr [rax], 0
	je .LBB216_9
.Ltmp23828:
	mov rdi, qword ptr [rsp + 104]
.Ltmp23829:
	call qword ptr [rip + mi_free@GOTPCREL]
	jmp .LBB216_9
.Ltmp23830:
.Ltmp23530:
	jmp .LBB216_8
.Ltmp23516:
	jmp .LBB216_75
.Ltmp23533:
.LBB216_75:
	mov r15, rax
	test rbx, rbx
	je .LBB216_9
	mov rdi, qword ptr [rsp + 8]
	call qword ptr [rip + mi_free@GOTPCREL]
	jmp .LBB216_9
.Ltmp23835:
.Ltmp23519:
	mov r15, rax
.Ltmp23836:
	test rbx, rbx
	je .LBB216_9
.Ltmp23837:
	mov rdi, r14
.Ltmp23838:
	call qword ptr [rip + mi_free@GOTPCREL]
	jmp .LBB216_9
.Ltmp23839:
.Ltmp23536:
	mov r15, rax
.Ltmp23840:
	test r14, r14
.Ltmp23841:
	mov r14, qword ptr [rsp]
.Ltmp23842:
	je .LBB216_10
.Ltmp23843:
	mov rdi, rbx
	call qword ptr [rip + mi_free@GOTPCREL]
.Ltmp23844:
	jmp .LBB216_10
.Ltmp23845:
.Ltmp23542:
	jmp .LBB216_8
.Ltmp23539:
.LBB216_8:
	mov r15, rax
.LBB216_9:
	mov r14, qword ptr [rsp]
.Ltmp23848:
.LBB216_10:
	mov rdi, r14
	call core::ptr::drop_in_place<masstree::leaf15::LeafNode15<masstree::value::LeafValue<u64>>>
.Ltmp23849:
	mov rdi, r14
	call qword ptr [rip + mi_free@GOTPCREL]
	mov rdi, r15
	call _Unwind_Resume@PLT
.Ltmp23850:
.Lfunc_end216:
	.size	<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated, .Lfunc_end216-<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated
	.cfi_endproc
.section ".gcc_except_table.<masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated","a",@progbits
	.p2align	2, 0x0
