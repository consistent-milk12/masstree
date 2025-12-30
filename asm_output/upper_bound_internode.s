masstree::ksearch::upper_bound_internode_generic:
.Lfunc_begin161:
	.cfi_startproc
	movzx ecx, byte ptr [rsi + 4]
.Ltmp17402:
	test cl, cl
	je .LBB161_32
.Ltmp17403:
	mov rax, qword ptr [rsi + 16]
.Ltmp17404:
	cmp rdi, rax
	jae .LBB161_2
.Ltmp17405:
.LBB161_32:
	xor eax, eax
.LBB161_33:
	ret
.Ltmp17407:
.LBB161_2:
	sete al
.Ltmp17408:
	cmp cl, 1
	sete dl
	or dl, al
	mov eax, 1
	jne .LBB161_33
.Ltmp17409:
	mov rdx, qword ptr [rsi + 24]
.Ltmp17410:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17411:
	sete al
.Ltmp17412:
	cmp cl, 2
	sete dl
.Ltmp17413:
	or dl, al
	mov eax, 2
	jne .LBB161_33
.Ltmp17414:
	mov rdx, qword ptr [rsi + 32]
.Ltmp17415:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17416:
	sete al
.Ltmp17417:
	cmp cl, 3
	sete dl
.Ltmp17418:
	or dl, al
	mov eax, 3
	jne .LBB161_33
.Ltmp17419:
	mov rdx, qword ptr [rsi + 40]
.Ltmp17420:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17421:
	sete al
.Ltmp17422:
	cmp cl, 4
	sete dl
.Ltmp17423:
	or dl, al
	mov eax, 4
	jne .LBB161_33
.Ltmp17424:
	mov rdx, qword ptr [rsi + 48]
.Ltmp17425:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17426:
	sete al
.Ltmp17427:
	cmp cl, 5
	sete dl
.Ltmp17428:
	or dl, al
	mov eax, 5
	jne .LBB161_33
.Ltmp17429:
	mov rdx, qword ptr [rsi + 56]
.Ltmp17430:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17431:
	sete al
.Ltmp17432:
	cmp cl, 6
	sete dl
.Ltmp17433:
	or dl, al
	mov eax, 6
	jne .LBB161_33
.Ltmp17434:
	mov rdx, qword ptr [rsi + 64]
.Ltmp17435:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17436:
	sete al
.Ltmp17437:
	cmp cl, 7
	sete dl
.Ltmp17438:
	or dl, al
	mov eax, 7
	jne .LBB161_33
.Ltmp17439:
	mov rdx, qword ptr [rsi + 72]
.Ltmp17440:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17441:
	sete al
.Ltmp17442:
	cmp cl, 8
	sete dl
.Ltmp17443:
	or dl, al
	mov eax, 8
	jne .LBB161_33
.Ltmp17444:
	mov rdx, qword ptr [rsi + 80]
.Ltmp17445:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17446:
	sete al
.Ltmp17447:
	cmp cl, 9
	sete dl
.Ltmp17448:
	or dl, al
	mov eax, 9
	jne .LBB161_33
.Ltmp17449:
	mov rdx, qword ptr [rsi + 88]
.Ltmp17450:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17451:
	sete al
.Ltmp17452:
	cmp cl, 10
	sete dl
.Ltmp17453:
	or dl, al
	mov eax, 10
	jne .LBB161_33
.Ltmp17454:
	mov rdx, qword ptr [rsi + 96]
.Ltmp17455:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17456:
	sete al
.Ltmp17457:
	cmp cl, 11
	sete dl
.Ltmp17458:
	or dl, al
	mov eax, 11
	jne .LBB161_33
.Ltmp17459:
	mov rdx, qword ptr [rsi + 104]
.Ltmp17460:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17461:
	sete al
.Ltmp17462:
	cmp cl, 12
	sete dl
.Ltmp17463:
	or dl, al
	mov eax, 12
	jne .LBB161_33
.Ltmp17464:
	mov rdx, qword ptr [rsi + 112]
.Ltmp17465:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17466:
	sete al
.Ltmp17467:
	cmp cl, 13
	sete dl
.Ltmp17468:
	or dl, al
	mov eax, 13
	jne .LBB161_33
.Ltmp17469:
	mov rdx, qword ptr [rsi + 120]
.Ltmp17470:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17471:
	sete al
.Ltmp17472:
	cmp cl, 14
	sete dl
.Ltmp17473:
	or dl, al
	mov eax, 14
	jne .LBB161_33
.Ltmp17474:
	mov rdx, qword ptr [rsi + 128]
.Ltmp17475:
	cmp rdi, rdx
	jb .LBB161_33
.Ltmp17476:
	setne dl
.Ltmp17477:
	cmp cl, 15
	setne cl
.Ltmp17478:
	mov eax, 15
	test dl, cl
	je .LBB161_33
.Ltmp17479:
	push rax
	.cfi_def_cfa_offset 16
.Ltmp17480:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.207]
	mov edi, 15
.Ltmp17481:
	mov esi, 15
.Ltmp17482:
	call core::panicking::panic_bounds_check
.Ltmp17483:
.Lfunc_end161:
	.size	masstree::ksearch::upper_bound_internode_generic, .Lfunc_end161-masstree::ksearch::upper_bound_internode_generic
	.cfi_endproc

.section ".text.core::ptr::drop_in_place<seize::guard::LocalGuard>","ax",@progbits
	.p2align	4
.type	core::ptr::drop_in_place<seize::guard::LocalGuard>,@function
