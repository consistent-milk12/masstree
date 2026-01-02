mttest:     file format elf64-x86-64


Disassembly of section .text:

000000000000d2c0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0>:
    d2c0:	push   r14
    d2c2:	push   r13
    d2c4:	push   r12
    d2c6:	mov    r12,rdi
    d2c9:	push   rbp
    d2ca:	mov    ebp,0x8
    d2cf:	push   rbx
    d2d0:	sub    rsp,0x30
    d2d4:	mov    r13,QWORD PTR [rdi+0x40]
    d2d8:	mov    rax,QWORD PTR fs:0x28
    d2e1:	mov    QWORD PTR [rsp+0x28],rax
    d2e6:	xor    eax,eax
    d2e8:	mov    QWORD PTR [rsp+0x10],r13
    d2ed:	mov    eax,DWORD PTR [r13+0x0]
    d2f1:	mov    rdx,r13
    d2f4:	jmp    d304 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x44>
    d2f6:	cs nop WORD PTR [rax+rax*1+0x0]
    d300:	pause
    d302:	mov    eax,DWORD PTR [rdx]
    d304:	test   al,0x6
    d306:	jne    d300 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x40>
    d308:	mov    DWORD PTR [rsp+0x8],eax
    d30c:	test   eax,0x40000000
    d311:	jne    d349 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x89>
    d313:	mov    rcx,QWORD PTR [rsp+0x10]
    d318:	mov    eax,DWORD PTR [rcx]
    d31a:	test   eax,eax
    d31c:	js     d338 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x78>
    d31e:	mov    rdx,QWORD PTR [rcx+0x108]
    d325:	test   rdx,rdx
    d328:	je     d344 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x84>
    d32a:	mov    eax,DWORD PTR [rdx]
    d32c:	mov    QWORD PTR [rsp+0x10],rdx
    d331:	jmp    d304 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x44>
    d333:	nop    DWORD PTR [rax+rax*1+0x0]
    d338:	mov    rdx,QWORD PTR [rcx+0x128]
    d33f:	test   rdx,rdx
    d342:	jne    d32a <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x6a>
    d344:	mov    rdx,rcx
    d347:	jmp    d32c <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x6c>
    d349:	xor    r8d,r8d
    d34c:	xor    edi,edi
    d34e:	mov    rbx,QWORD PTR [rsp+rdi*8+0x10]
    d353:	test   eax,eax
    d355:	js     d40b <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x14b>
    d35b:	nop    DWORD PTR [rax+rax*1+0x0]
    d360:	prefetcht0 BYTE PTR [rbx+0x40]
    d364:	prefetcht0 BYTE PTR [rbx+0x80]
    d36b:	prefetcht0 BYTE PTR [rbx+0xc0]
    d372:	movzx  ecx,BYTE PTR [rbx+0x4]
    d376:	xor    edx,edx
    d378:	cmp    ecx,edx
    d37a:	jle    d3b0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0xf0>
    d37c:	lea    eax,[rcx+rdx*1]
    d37f:	mov    r9,QWORD PTR [r12+0x8]
    d384:	sar    eax,1
    d386:	movsxd rsi,eax
    d389:	mov    rsi,QWORD PTR [rbx+rsi*8+0x10]
    d38e:	cmp    rsi,r9
    d391:	je     d518 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x258>
    d397:	cmp    r9,rsi
    d39a:	jb     d520 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x260>
    d3a0:	lea    edx,[rax+0x1]
    d3a3:	cmp    ecx,edx
    d3a5:	jg     d37c <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0xbc>
    d3a7:	nop    WORD PTR [rax+rax*1+0x0]
    d3b0:	mov    ecx,r8d
    d3b3:	movsxd rdx,edx
    d3b6:	mov    rdx,QWORD PTR [rbx+rdx*8+0x88]
    d3be:	xor    ecx,0x1
    d3c1:	mov    esi,ecx
    d3c3:	mov    QWORD PTR [rsp+rsi*8+0x10],rdx
    d3c8:	test   rdx,rdx
    d3cb:	jne    d3da <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x11a>
    d3cd:	jmp    d2e8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x28>
    d3d2:	nop    WORD PTR [rax+rax*1+0x0]
    d3d8:	pause
    d3da:	mov    eax,DWORD PTR [rdx]
    d3dc:	test   al,0x6
    d3de:	jne    d3d8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x118>
    d3e0:	mov    DWORD PTR [rsp+rsi*4+0x8],eax
    d3e4:	mov    edx,DWORD PTR [rsp+rdi*4+0x8]
    d3e8:	mov    eax,DWORD PTR [rbx]
    d3ea:	xor    edx,eax
    d3ec:	cmp    edx,0x1
    d3ef:	ja     d530 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x270>
    d3f5:	mov    eax,DWORD PTR [rsp+rsi*4+0x8]
    d3f9:	mov    edi,ecx
    d3fb:	mov    rbx,QWORD PTR [rsp+rdi*8+0x10]
    d400:	mov    r8,rdi
    d403:	test   eax,eax
    d405:	jns    d360 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0xa0>
    d40b:	mov    r14d,DWORD PTR [rsp+rdi*4+0x8]
    d410:	mov    QWORD PTR [r12],rbx
    d414:	mov    DWORD PTR [r12+0x28],r14d
    d419:	test   r14d,0x20000000
    d420:	jne    d2e8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x28>
    d426:	cs nop WORD PTR [rax+rax*1+0x0]
    d430:	prefetcht0 BYTE PTR [rbx+0x40]
    d434:	prefetcht0 BYTE PTR [rbx+0x80]
    d43b:	prefetcht0 BYTE PTR [rbx+0xc0]
    d442:	cmp    BYTE PTR [rbx+0x4],0x0
    d446:	jle    d798 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x4d8>
    d44c:	prefetcht0 BYTE PTR [rbx+0x138]
    d453:	mov    rdi,QWORD PTR [rbx+0x18]
    d457:	mov    r8,QWORD PTR [r12+0x8]
    d45c:	mov    r9d,DWORD PTR [r12+0x10]
    d461:	mov    edx,edi
    d463:	mov    QWORD PTR [r12+0x30],rdi
    d468:	and    edx,0xf
    d46b:	je     d4b7 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1f7>
    d46d:	xor    r10d,r10d
    d470:	cmp    r9d,0x8
    d474:	jg     d575 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2b5>
    d47a:	nop    WORD PTR [rax+rax*1+0x0]
    d480:	lea    esi,[rdx+r10*1]
    d484:	mov    rax,rdi
    d487:	sar    esi,1
    d489:	lea    ecx,[rsi*4+0x4]
    d490:	shr    rax,cl
    d493:	and    eax,0xf
    d496:	movzx  ecx,BYTE PTR [rbx+rax*1+0x6]
    d49b:	cmp    r8,QWORD PTR [rbx+rax*8+0x20]
    d4a0:	je     d688 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x3c8>
    d4a6:	jb     d5b0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2f0>
    d4ac:	lea    r10d,[rsi+0x1]
    d4b0:	cmp    edx,r10d
    d4b3:	jg     d480 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1c0>
    d4b5:	xor    edx,edx
    d4b7:	xor    r14d,DWORD PTR [rbx]
    d4ba:	cmp    r14d,0x1
    d4be:	jbe    d736 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x476>
    d4c4:	mov    rbx,QWORD PTR [r12]
    d4c8:	mov    edx,DWORD PTR [r12+0x28]
    d4cd:	mov    eax,DWORD PTR [rbx]
    d4cf:	test   al,0x6
    d4d1:	je     d4e0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x220>
    d4d3:	nop    DWORD PTR [rax+rax*1+0x0]
    d4d8:	pause
    d4da:	mov    eax,DWORD PTR [rbx]
    d4dc:	test   al,0x6
    d4de:	jne    d4d8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x218>
    d4e0:	mov    DWORD PTR [r12+0x28],eax
    d4e5:	mov    r14d,DWORD PTR [r12+0x28]
    d4ea:	mov    eax,edx
    d4ec:	xor    eax,r14d
    d4ef:	mov    edx,r14d
    d4f2:	and    edx,0x20000000
    d4f8:	cmp    eax,0x1ff
    d4fd:	ja     d5ca <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x30a>
    d503:	mov    QWORD PTR [r12],rbx
    d507:	test   edx,edx
    d509:	je     d430 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x170>
    d50f:	jmp    d2e8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x28>
    d514:	nop    DWORD PTR [rax+0x0]
    d518:	lea    edx,[rax+0x1]
    d51b:	jmp    d3b0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0xf0>
    d520:	mov    ecx,eax
    d522:	jmp    d378 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0xb8>
    d527:	nop    WORD PTR [rax+rax*1+0x0]
    d530:	mov    edx,DWORD PTR [rsp+rdi*4+0x8]
    d534:	test   al,0x6
    d536:	je     d548 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x288>
    d538:	nop    DWORD PTR [rax+rax*1+0x0]
    d540:	pause
    d542:	mov    eax,DWORD PTR [rbx]
    d544:	test   al,0x6
    d546:	jne    d540 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x280>
    d548:	mov    DWORD PTR [rsp+rdi*4+0x8],eax
    d54c:	xor    eax,edx
    d54e:	cmp    eax,0x1ff
    d553:	ja     d86a <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x5aa>
    d559:	mov    eax,DWORD PTR [rsp+rdi*4+0x8]
    d55d:	jmp    d34e <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x8e>
    d562:	nop    WORD PTR [rax+rax*1+0x0]
    d568:	jae    d5a0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2e0>
    d56a:	mov    edx,esi
    d56c:	cmp    r10d,edx
    d56f:	jge    d4b5 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1f5>
    d575:	lea    esi,[r10+rdx*1]
    d579:	mov    rax,rdi
    d57c:	sar    esi,1
    d57e:	lea    ecx,[rsi*4+0x4]
    d585:	shr    rax,cl
    d588:	and    eax,0xf
    d58b:	movzx  ecx,BYTE PTR [rbx+rax*1+0x6]
    d590:	cmp    r8,QWORD PTR [rbx+rax*8+0x20]
    d595:	jne    d568 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2a8>
    d597:	cmp    cl,0x8
    d59a:	ja     d697 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x3d7>
    d5a0:	lea    r10d,[rsi+0x1]
    d5a4:	jmp    d56c <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2ac>
    d5a6:	cs nop WORD PTR [rax+rax*1+0x0]
    d5b0:	mov    edx,esi
    d5b2:	jmp    d4b0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1f0>
    d5b7:	mov    r14d,edx
    d5ba:	test   dl,0x6
    d5bd:	je     d5ca <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x30a>
    d5bf:	pause
    d5c1:	mov    r14d,DWORD PTR [rbx]
    d5c4:	test   r14b,0x6
    d5c8:	jne    d5bf <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2ff>
    d5ca:	mov    rax,QWORD PTR [rbx+0x18]
    d5ce:	mov    ecx,0x4
    d5d3:	mov    edx,eax
    d5d5:	and    edx,0xf
    d5d8:	je     d5e1 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x321>
    d5da:	lea    ecx,[rdx*4+0x0]
    d5e1:	shr    rax,cl
    d5e4:	mov    rdx,QWORD PTR [r12+0x8]
    d5e9:	and    eax,0xf
    d5ec:	movzx  ecx,BYTE PTR [rbx+rax*1+0x6]
    d5f1:	mov    rax,QWORD PTR [rbx+rax*8+0x20]
    d5f6:	cmp    rax,rdx
    d5f9:	je     d853 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x593>
    d5ff:	cmp    rdx,rax
    d602:	sbb    eax,eax
    d604:	or     eax,0x1
    d607:	mov    edx,DWORD PTR [rbx]
    d609:	xor    r14d,edx
    d60c:	cmp    r14d,0x1
    d610:	ja     d5b7 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2f7>
    d612:	mov    r14d,DWORD PTR [r12+0x28]
    d617:	mov    edx,r14d
    d61a:	and    edx,0x20000000
    d620:	test   eax,eax
    d622:	jle    d503 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x243>
    d628:	test   edx,edx
    d62a:	jne    d679 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x3b9>
    d62c:	mov    rax,QWORD PTR [rbx+0x118]
    d633:	mov    rdx,rbx
    d636:	mov    rbx,rax
    d639:	and    rbx,0xfffffffffffffffe
    d63d:	cmp    rax,0x1
    d641:	jbe    d847 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x587>
    d647:	mov    rax,QWORD PTR [rbx+0x20]
    d64b:	cmp    QWORD PTR [r12+0x8],rax
    d650:	jae    d662 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x3a2>
    d652:	jmp    d847 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x587>
    d657:	nop    WORD PTR [rax+rax*1+0x0]
    d660:	pause
    d662:	mov    r14d,DWORD PTR [rbx]
    d665:	test   r14b,0x6
    d669:	jne    d660 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x3a0>
    d66b:	mov    DWORD PTR [r12+0x28],r14d
    d670:	test   r14d,0x20000000
    d677:	je     d62c <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x36c>
    d679:	mov    QWORD PTR [r12],rbx
    d67d:	jmp    d2e8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x28>
    d682:	nop    WORD PTR [rax+rax*1+0x0]
    d688:	cmp    r9d,ecx
    d68b:	js     d5b0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x2f0>
    d691:	jne    d4ac <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1ec>
    d697:	mov    rdx,QWORD PTR [rbx+rax*8+0x98]
    d69f:	mov    QWORD PTR [r12+0x38],rdx
    d6a4:	movzx  ecx,BYTE PTR [rbx+rax*1+0x6]
    d6a9:	test   cl,cl
    d6ab:	js     d7b8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x4f8>
    d6b1:	prefetcht0 BYTE PTR [rdx]
    d6b4:	cmp    ecx,0x3f
    d6b7:	jle    d825 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x565>
    d6bd:	cmp    ecx,0x40
    d6c0:	jne    d7d6 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x516>
    d6c6:	mov    rdx,QWORD PTR [rbx+0x110]
    d6cd:	test   rdx,rdx
    d6d0:	je     d808 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x548>
    d6d6:	movzx  ecx,WORD PTR [rdx+rax*4+0x6]
    d6db:	movzx  eax,WORD PTR [rdx+rax*4+0x4]
    d6e0:	lea    rdi,[rdx+rax*1]
    d6e4:	sub    r9d,0x8
    d6e8:	xor    edx,edx
    d6ea:	cmp    r9d,ecx
    d6ed:	jne    d4b7 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1f7>
    d6f3:	mov    rax,QWORD PTR [r12+0x18]
    d6f8:	cmp    ecx,0x8
    d6fb:	jg     d82f <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x56f>
    d701:	mov    rdx,QWORD PTR [rdi]
    d704:	xor    rdx,QWORD PTR [rax+0x8]
    d708:	mov    rax,rdx
    d70b:	test   ecx,ecx
    d70d:	je     d825 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x565>
    d713:	mov    edx,ebp
    d715:	sub    edx,ecx
    d717:	lea    ecx,[rdx*8+0x0]
    d71e:	xor    edx,edx
    d720:	shl    rax,cl
    d723:	test   rax,rax
    d726:	sete   dl
    d729:	xor    r14d,DWORD PTR [rbx]
    d72c:	cmp    r14d,0x1
    d730:	ja     d4c4 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x204>
    d736:	test   edx,edx
    d738:	jns    d8cc <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x60c>
    d73e:	mov    eax,DWORD PTR [r12+0x10]
    d743:	mov    rsi,QWORD PTR [r12+0x18]
    d748:	lea    edx,[rax-0x8]
    d74b:	lea    rcx,[rsi+0x8]
    d74f:	xor    eax,eax
    d751:	mov    QWORD PTR [r12+0x18],rcx
    d756:	mov    DWORD PTR [r12+0x10],edx
    d75b:	test   edx,edx
    d75d:	jle    d782 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x4c2>
    d75f:	cmp    edx,0x7
    d762:	jg     d8c3 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x603>
    d768:	mov    eax,ebp
    d76a:	sub    eax,edx
    d76c:	movsxd rsi,eax
    d76f:	sub    rcx,rsi
    d772:	mov    rdx,rcx
    d775:	lea    ecx,[rax*8+0x0]
    d77c:	mov    rax,QWORD PTR [rdx]
    d77f:	shr    rax,cl
    d782:	bswap  rax
    d785:	mov    r13,QWORD PTR [r12+0x38]
    d78a:	mov    QWORD PTR [r12+0x8],rax
    d78f:	jmp    d2e8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x28>
    d794:	nop    DWORD PTR [rax+0x0]
    d798:	je     d453 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x193>
    d79e:	mov    rax,QWORD PTR [rbx+0x110]
    d7a5:	prefetcht0 BYTE PTR [rax]
    d7a8:	prefetcht0 BYTE PTR [rax+0x40]
    d7ac:	jmp    d453 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x193>
    d7b1:	nop    DWORD PTR [rax+0x0]
    d7b8:	prefetcht0 BYTE PTR [rdx]
    d7bb:	prefetcht0 BYTE PTR [rdx+0x40]
    d7bf:	prefetcht0 BYTE PTR [rdx+0x80]
    d7c6:	prefetcht0 BYTE PTR [rdx+0xc0]
    d7cd:	cmp    cl,0x80
    d7d0:	je     d8ee <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x62e>
    d7d6:	mov    rax,QWORD PTR [rsp+0x28]
    d7db:	sub    rax,QWORD PTR fs:0x28
    d7e4:	jne    d8f8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x638>
    d7ea:	xor    ecx,ecx
    d7ec:	lea    rdx,[rip+0x4e762]        # 5bf55 <_IO_stdin_used+0x1f55>
    d7f3:	mov    esi,0x18a
    d7f8:	lea    rdi,[rip+0x4e770]        # 5bf6f <_IO_stdin_used+0x1f6f>
    d7ff:	call   590f0 <_Z26fail_masstree_preconditionPKciS0_S0_>
    d804:	nop    DWORD PTR [rax+0x0]
    d808:	movzx  ecx,BYTE PTR [rbx+rax*2+0x13b]
    d810:	movzx  eax,BYTE PTR [rbx+rax*2+0x13a]
    d818:	lea    rdi,[rbx+rax*1+0x138]
    d820:	jmp    d6e4 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x424>
    d825:	mov    edx,0x1
    d82a:	jmp    d4b7 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1f7>
    d82f:	movsxd rdx,ecx
    d832:	lea    rsi,[rax+0x8]
    d836:	call   58a0 <memcmp@plt>
    d83b:	xor    edx,edx
    d83d:	test   eax,eax
    d83f:	sete   dl
    d842:	jmp    d4b7 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1f7>
    d847:	mov    QWORD PTR [r12],rdx
    d84b:	mov    rbx,rdx
    d84e:	jmp    d430 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x170>
    d853:	mov    eax,DWORD PTR [r12+0x10]
    d858:	cmp    eax,0x8
    d85b:	jle    d8bc <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x5fc>
    d85d:	xor    eax,eax
    d85f:	cmp    ecx,0x8
    d862:	setle  al
    d865:	jmp    d607 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x347>
    d86a:	mov    ecx,DWORD PTR [rsp+rdi*4+0x8]
    d86e:	movzx  eax,BYTE PTR [rbx+0x4]
    d872:	mov    edx,0x1
    d877:	test   eax,eax
    d879:	je     d899 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x5d9>
    d87b:	sub    eax,0x1
    d87e:	mov    rsi,QWORD PTR [r12+0x8]
    d883:	xor    edx,edx
    d885:	cdqe
    d887:	mov    rax,QWORD PTR [rbx+rax*8+0x10]
    d88c:	cmp    rax,rsi
    d88f:	je     d899 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x5d9>
    d891:	cmp    rsi,rax
    d894:	sbb    edx,edx
    d896:	or     edx,0x1
    d899:	mov    eax,DWORD PTR [rbx]
    d89b:	xor    ecx,eax
    d89d:	cmp    ecx,0x1
    d8a0:	ja     d8b4 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x5f4>
    d8a2:	cmp    edx,0x1
    d8a5:	jne    d559 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x299>
    d8ab:	jmp    d2e8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x28>
    d8b0:	pause
    d8b2:	mov    eax,DWORD PTR [rbx]
    d8b4:	test   al,0x6
    d8b6:	jne    d8b0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x5f0>
    d8b8:	mov    ecx,eax
    d8ba:	jmp    d86e <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x5ae>
    d8bc:	sub    eax,ecx
    d8be:	jmp    d607 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x347>
    d8c3:	mov    rax,QWORD PTR [rsi+0x8]
    d8c7:	jmp    d782 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x4c2>
    d8cc:	mov    eax,edx
    d8ce:	and    eax,0x1
    d8d1:	mov    rdx,QWORD PTR [rsp+0x28]
    d8d6:	sub    rdx,QWORD PTR fs:0x28
    d8df:	jne    d8f8 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x638>
    d8e1:	add    rsp,0x30
    d8e5:	pop    rbx
    d8e6:	pop    rbp
    d8e7:	pop    r12
    d8e9:	pop    r13
    d8eb:	pop    r14
    d8ed:	ret
    d8ee:	mov    edx,0xfffffff8
    d8f3:	jmp    d4b7 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0+0x1f7>
    d8f8:	call   5ac0 <__stack_chk_fail@plt>
    d8fd:	nop
    d8fe:	xchg   ax,ax

000000000000d900 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0>:
    d900:	push   r12
    d902:	movq   xmm0,rsi
    d907:	mov    r12,rdi
    d90a:	push   rbp
    d90b:	punpcklqdq xmm0,xmm0
    d90f:	mov    rbp,rcx
    d912:	push   rbx
    d913:	mov    rbx,rdx
    d916:	sub    rsp,0x70
    d91a:	mov    rax,QWORD PTR fs:0x28
    d923:	mov    QWORD PTR [rsp+0x68],rax
    d928:	xor    eax,eax
    d92a:	test   edx,edx
    d92c:	jle    d936 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0x36>
    d92e:	cmp    edx,0x7
    d931:	jle    d9a0 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0xa0>
    d933:	mov    rax,QWORD PTR [rsi]
    d936:	bswap  rax
    d939:	mov    QWORD PTR [rsp+0x18],rax
    d93e:	mov    rax,QWORD PTR [r12]
    d942:	mov    DWORD PTR [rsp+0x20],ebx
    d946:	mov    edx,DWORD PTR [rax]
    d948:	movups XMMWORD PTR [rsp+0x28],xmm0
    d94d:	mov    QWORD PTR [rsp+0x48],0x0
    d956:	test   edx,0x40000000
    d95c:	je     da00 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0x100>
    d962:	lea    rdi,[rsp+0x10]
    d967:	mov    QWORD PTR [rsp+0x50],rax
    d96c:	call   d2c0 <_ZN8Masstree16unlocked_tcursorINS_26default_query_table_paramsEE13find_unlockedER10threadinfo.isra.0>
    d971:	test   al,al
    d973:	je     d981 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0x81>
    d975:	mov    rdx,QWORD PTR [rsp+0x48]
    d97a:	test   BYTE PTR [rdx],0x1
    d97d:	je     d9d0 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0xd0>
    d97f:	xor    eax,eax
    d981:	mov    rdx,QWORD PTR [rsp+0x68]
    d986:	sub    rdx,QWORD PTR fs:0x28
    d98f:	jne    da29 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0x129>
    d995:	add    rsp,0x70
    d999:	pop    rbx
    d99a:	pop    rbp
    d99b:	pop    r12
    d99d:	ret
    d99e:	xchg   ax,ax
    d9a0:	lea    rdi,[rsp+0x60]
    d9a5:	movsxd rdx,edx
    d9a8:	movaps XMMWORD PTR [rsp],xmm0
    d9ac:	mov    QWORD PTR [rsp+0x60],0x0
    d9b5:	call   59e0 <memcpy@plt>
    d9ba:	mov    rax,QWORD PTR [rsp+0x60]
    d9bf:	movdqa xmm0,XMMWORD PTR [rsp]
    d9c4:	jmp    d936 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0x36>
    d9c9:	nop    DWORD PTR [rax+0x0]
    d9d0:	cmp    WORD PTR [rdx+0x8],0x0
    d9d5:	jne    d9e8 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0xe8>
    d9d7:	xor    ecx,ecx
    d9d9:	xor    edx,edx
    d9db:	mov    QWORD PTR [rbp+0x0],rdx
    d9df:	mov    DWORD PTR [rbp+0x8],ecx
    d9e2:	jmp    d981 <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0x81>
    d9e4:	nop    DWORD PTR [rax+0x0]
    d9e8:	movzx  esi,WORD PTR [rdx+0xa]
    d9ec:	movzx  ecx,WORD PTR [rdx+0xc]
    d9f0:	sub    ecx,esi
    d9f2:	lea    rdx,[rdx+rsi*1+0x8]
    d9f7:	movzx  ecx,cx
    d9fa:	jmp    d9db <_ZN5queryI9value_bagItEE8run_get1IN8Masstree11basic_tableINS4_26default_query_table_paramsEEEEEbRT_N4lcdf3StrEiRSB_R10threadinfo.constprop.0.isra.0+0xdb>
    d9fc:	nop    DWORD PTR [rax+0x0]
