masstree::leaf24::LeafNode24<S>::wait_for_split:
.Lfunc_begin163:
	.cfi_startproc
	xor eax, eax
	jmp .LBB163_1
.Ltmp17494:
	.p2align	4
.LBB163_24:
	#MEMBARRIER
	inc rax
	cmp rax, 1001
	je .LBB163_25
.Ltmp17497:
.LBB163_1:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17498:
	test cl, 1
.Ltmp17499:
	je .LBB163_25
.Ltmp17500:
	pause
.Ltmp17501:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17502:
	test cl, 1
.Ltmp17503:
	je .LBB163_25
.Ltmp17504:
	pause
.Ltmp17505:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17506:
	test cl, 1
.Ltmp17507:
	je .LBB163_25
.Ltmp17508:
	pause
.Ltmp17509:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17510:
	test cl, 1
.Ltmp17511:
	je .LBB163_25
.Ltmp17512:
	pause
.Ltmp17513:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17514:
	test cl, 1
.Ltmp17515:
	je .LBB163_25
.Ltmp17516:
	pause
.Ltmp17517:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17518:
	test cl, 1
.Ltmp17519:
	je .LBB163_25
.Ltmp17520:
	pause
.Ltmp17521:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17522:
	test cl, 1
.Ltmp17523:
	je .LBB163_25
.Ltmp17524:
	pause
.Ltmp17525:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17526:
	test cl, 1
.Ltmp17527:
	je .LBB163_25
.Ltmp17528:
	pause
.Ltmp17529:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17530:
	test cl, 1
.Ltmp17531:
	je .LBB163_25
.Ltmp17532:
	pause
.Ltmp17533:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17534:
	test cl, 1
.Ltmp17535:
	je .LBB163_25
.Ltmp17536:
	pause
.Ltmp17537:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17538:
	test cl, 1
.Ltmp17539:
	je .LBB163_25
.Ltmp17540:
	pause
.Ltmp17541:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17542:
	test cl, 1
.Ltmp17543:
	je .LBB163_25
.Ltmp17544:
	pause
.Ltmp17545:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17546:
	test cl, 1
.Ltmp17547:
	je .LBB163_25
.Ltmp17548:
	pause
.Ltmp17549:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17550:
	test cl, 1
.Ltmp17551:
	je .LBB163_25
.Ltmp17552:
	pause
.Ltmp17553:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17554:
	test cl, 1
.Ltmp17555:
	je .LBB163_25
.Ltmp17556:
	pause
.Ltmp17557:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17558:
	test cl, 1
.Ltmp17559:
	je .LBB163_25
.Ltmp17560:
	pause
.Ltmp17561:
	mov rcx, qword ptr [rdi + 544]
.Ltmp17562:
	test cl, 1
.Ltmp17563:
	je .LBB163_25
.Ltmp17564:
	mov ecx, dword ptr [rdi]
	test cl, 6
	je .LBB163_24
.Ltmp17565:
	xor ecx, ecx
	jmp .LBB163_20
	.p2align	4
.LBB163_23:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov edx, dword ptr [rdi]
	test dl, 6
	je .LBB163_24
.LBB163_20:
	xor edx, edx
	.p2align	4
.LBB163_21:
	mov esi, edx
	pause
	cmp edx, ecx
	adc edx, 0
	cmp esi, ecx
	jae .LBB163_23
	cmp edx, ecx
	jbe .LBB163_21
	jmp .LBB163_23
.Ltmp17570:
.LBB163_25:
	ret
.Lfunc_end163:
	.size	masstree::leaf24::LeafNode24<S>::wait_for_split, .Lfunc_end163-masstree::leaf24::LeafNode24<S>::wait_for_split
	.cfi_endproc

.section .text.core::ops::function::FnOnce::call_once,"ax",@progbits
	.p2align	4
.type	core::ops::function::FnOnce::call_once,@function
core::ops::function::FnOnce::call_once:
masstree::leaf15::LeafNode15<S>::wait_for_split:
.Lfunc_begin217:
	.cfi_startproc
	xor eax, eax
	jmp .LBB217_1
.Ltmp23851:
	.p2align	4
.LBB217_24:
	#MEMBARRIER
	inc rax
	cmp rax, 1001
	je .LBB217_25
.Ltmp23854:
.LBB217_1:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23855:
	test cl, 1
.Ltmp23856:
	je .LBB217_25
.Ltmp23857:
	pause
.Ltmp23858:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23859:
	test cl, 1
.Ltmp23860:
	je .LBB217_25
.Ltmp23861:
	pause
.Ltmp23862:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23863:
	test cl, 1
.Ltmp23864:
	je .LBB217_25
.Ltmp23865:
	pause
.Ltmp23866:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23867:
	test cl, 1
.Ltmp23868:
	je .LBB217_25
.Ltmp23869:
	pause
.Ltmp23870:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23871:
	test cl, 1
.Ltmp23872:
	je .LBB217_25
.Ltmp23873:
	pause
.Ltmp23874:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23875:
	test cl, 1
.Ltmp23876:
	je .LBB217_25
.Ltmp23877:
	pause
.Ltmp23878:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23879:
	test cl, 1
.Ltmp23880:
	je .LBB217_25
.Ltmp23881:
	pause
.Ltmp23882:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23883:
	test cl, 1
.Ltmp23884:
	je .LBB217_25
.Ltmp23885:
	pause
.Ltmp23886:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23887:
	test cl, 1
.Ltmp23888:
	je .LBB217_25
.Ltmp23889:
	pause
.Ltmp23890:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23891:
	test cl, 1
.Ltmp23892:
	je .LBB217_25
.Ltmp23893:
	pause
.Ltmp23894:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23895:
	test cl, 1
.Ltmp23896:
	je .LBB217_25
.Ltmp23897:
	pause
.Ltmp23898:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23899:
	test cl, 1
.Ltmp23900:
	je .LBB217_25
.Ltmp23901:
	pause
.Ltmp23902:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23903:
	test cl, 1
.Ltmp23904:
	je .LBB217_25
.Ltmp23905:
	pause
.Ltmp23906:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23907:
	test cl, 1
.Ltmp23908:
	je .LBB217_25
.Ltmp23909:
	pause
.Ltmp23910:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23911:
	test cl, 1
.Ltmp23912:
	je .LBB217_25
.Ltmp23913:
	pause
.Ltmp23914:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23915:
	test cl, 1
.Ltmp23916:
	je .LBB217_25
.Ltmp23917:
	pause
.Ltmp23918:
	mov rcx, qword ptr [rdi + 328]
.Ltmp23919:
	test cl, 1
.Ltmp23920:
	je .LBB217_25
.Ltmp23921:
	mov ecx, dword ptr [rdi]
	test cl, 6
	je .LBB217_24
.Ltmp23922:
	xor ecx, ecx
	jmp .LBB217_20
	.p2align	4
.LBB217_23:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov edx, dword ptr [rdi]
	test dl, 6
	je .LBB217_24
.LBB217_20:
	xor edx, edx
	.p2align	4
.LBB217_21:
	mov esi, edx
	pause
	cmp edx, ecx
	adc edx, 0
	cmp esi, ecx
	jae .LBB217_23
	cmp edx, ecx
	jbe .LBB217_21
	jmp .LBB217_23
.Ltmp23927:
.LBB217_25:
	ret
.Lfunc_end217:
	.size	masstree::leaf15::LeafNode15<S>::wait_for_split, .Lfunc_end217-masstree::leaf15::LeafNode15<S>::wait_for_split
	.cfi_endproc

.section ".text.core::ptr::drop_in_place<seize::collector::Collector>","ax",@progbits
	.p2align	4
.type	core::ptr::drop_in_place<seize::collector::Collector>,@function
