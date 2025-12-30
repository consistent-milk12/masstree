mttest:     file format elf64-x86-64


Disassembly of section .text:

000000000001e3d0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo>:
   1e3d0:	endbr64
   1e3d4:	push   r15
   1e3d6:	push   r14
   1e3d8:	mov    r14,rdi
   1e3db:	push   r13
   1e3dd:	push   r12
   1e3df:	push   rbp
   1e3e0:	push   rbx
   1e3e1:	sub    rsp,0x98
   1e3e8:	mov    QWORD PTR [rsp+0x10],rsi
   1e3ed:	mov    rax,QWORD PTR fs:0x28
   1e3f6:	mov    QWORD PTR [rsp+0x88],rax
   1e3fe:	mov    rax,QWORD PTR [rdi]
   1e401:	mov    rdi,QWORD PTR [rax+0x18]
   1e405:	mov    edx,edi
   1e407:	and    edx,0xf
   1e40a:	cmp    edx,0xf
   1e40d:	je     1e44d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x7d>
   1e40f:	lea    ecx,[rdx*4+0x0]
   1e416:	mov    rsi,rdi
   1e419:	mov    rdx,rdi
   1e41c:	shr    rsi,cl
   1e41f:	shr    rdx,0x38
   1e423:	xor    rdx,rsi
   1e426:	and    edx,0xf0
   1e42c:	mov    rsi,rdx
   1e42f:	shl    rdx,cl
   1e432:	shl    rsi,0x38
   1e436:	or     rdx,rsi
   1e439:	xor    rdx,rdi
   1e43c:	mov    rcx,rdx
   1e43f:	shr    rcx,0x3c
   1e443:	mov    DWORD PTR [r14+0x2c],ecx
   1e447:	jne    1f600 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1230>
   1e44d:	mov    rdx,QWORD PTR [rax+0x110]
   1e454:	mov    r12,QWORD PTR [rax+0x130]
   1e45b:	test   rdx,rdx
   1e45e:	je     1f396 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xfc6>
   1e464:	movzx  eax,WORD PTR [rdx]
   1e467:	cmp    rax,0x80
   1e46d:	ja     1f2ec <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xf1c>
   1e473:	lea    edx,[rax+0x138]
   1e479:	lea    rbp,[rax+0x138]
   1e480:	and    edx,0x3f
   1e483:	jne    1f6fb <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x132b>
   1e489:	lea    rax,[rbp+0x3f]
   1e48d:	xor    r13d,r13d
   1e490:	shr    rax,0x6
   1e494:	mov    esi,eax
   1e496:	sub    eax,0x1
   1e499:	mov    rdi,QWORD PTR [rsp+0x10]
   1e49e:	movsxd rbx,eax
   1e4a1:	lea    rax,[rbx+0x8]
   1e4a5:	mov    r15,QWORD PTR [rdi+rax*8]
   1e4a9:	test   r15,r15
   1e4ac:	je     1f3ac <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xfdc>
   1e4b2:	mov    rax,QWORD PTR [r15]
   1e4b5:	mov    rdi,QWORD PTR [rsp+0x10]
   1e4ba:	mov    QWORD PTR [rdi+rbx*8+0x40],rax
   1e4bf:	movabs rax,0x123456789abcde0
   1e4c9:	mov    BYTE PTR [r15+0x5],0x0
   1e4ce:	mov    DWORD PTR [r15],0x80000000
   1e4d5:	mov    QWORD PTR [r15+0x18],rax
   1e4d9:	mov    QWORD PTR [r15+0x110],0x0
   1e4e4:	mov    QWORD PTR [r15+0x128],0x0
   1e4ef:	test   r13,r13
   1e4f2:	jne    1f8a7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x14d7>
   1e4f8:	mov    eax,ebp
   1e4fa:	sar    eax,0x6
   1e4fd:	sub    eax,0x5
   1e500:	mov    BYTE PTR [r15+0x4],al
   1e504:	test   al,al
   1e506:	jle    1e547 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x177>
   1e508:	sub    ebp,0x39
   1e50b:	xor    r8d,r8d
   1e50e:	mov    BYTE PTR [r15+0x138],0x20
   1e516:	pxor   xmm0,xmm0
   1e51a:	mov    BYTE PTR [r15+0x139],bpl
   1e521:	mov    QWORD PTR [r15+0x14a],0x0
   1e52c:	mov    DWORD PTR [r15+0x152],0x0
   1e537:	mov    WORD PTR [r15+0x156],r8w
   1e53f:	movups XMMWORD PTR [r15+0x13a],xmm0
   1e547:	mov    rdi,QWORD PTR [r14]
   1e54a:	mov    QWORD PTR [r15+0x130],r12
   1e551:	mov    rdx,r14
   1e554:	mov    rsi,r15
   1e557:	mov    r8,QWORD PTR [rsp+0x10]
   1e55c:	lea    rcx,[rsp+0x70]
   1e561:	mov    eax,DWORD PTR [rdi]
   1e563:	mov    DWORD PTR [r15],eax
   1e566:	call   1d1c0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE10split_intoEPS2_PNS_7tcursorIS1_EERmR10threadinfo>
   1e56b:	mov    rbp,QWORD PTR [r14]
   1e56e:	mov    DWORD PTR [rsp+0x1c],eax
   1e572:	mov    eax,DWORD PTR [rbp+0x0]
   1e575:	test   al,0x1
   1e577:	je     1f720 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1350>
   1e57d:	cmp    DWORD PTR [rsp+0x1c],0x2
   1e582:	mov    DWORD PTR [rsp+0x18],0x1
   1e58a:	sete   dl
   1e58d:	mov    DWORD PTR [rsp+0x8],0x0
   1e595:	cmp    dl,0x1
   1e598:	sbb    edi,edi
   1e59a:	and    edi,0xfffffff9
   1e59d:	add    edi,0xf
   1e5a0:	cmp    dl,0x1
   1e5a3:	mov    edx,0xf
   1e5a8:	sbb    r10d,r10d
   1e5ab:	mov    DWORD PTR [rsp+0x50],edi
   1e5af:	and    r10d,0xfffffff9
   1e5b3:	add    r10d,0xf
   1e5b7:	sub    edx,r10d
   1e5ba:	lea    edi,[r10-0x1]
   1e5be:	mov    BYTE PTR [rsp+0x57],r10b
   1e5c3:	mov    BYTE PTR [rsp+0x55],dl
   1e5c7:	mov    BYTE PTR [rsp+0x56],dil
   1e5cc:	nop    DWORD PTR [rax+0x0]
   1e5d0:	test   BYTE PTR [r15],0x1
   1e5d4:	je     1f720 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1350>
   1e5da:	test   eax,eax
   1e5dc:	js     1e660 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x290>
   1e5e2:	test   al,0x4
   1e5e4:	jne    1e640 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x270>
   1e5e6:	jmp    1f720 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1350>
   1e5eb:	nop    DWORD PTR [rax+rax*1+0x0]
   1e5f0:	mov    esi,ecx
   1e5f2:	mov    eax,ecx
   1e5f4:	or     esi,0x1
   1e5f7:	lock cmpxchg DWORD PTR [rbx],esi
   1e5fb:	jne    1e657 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x287>
   1e5fd:	test   cl,0x6
   1e600:	jne    1f681 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x12b1>
   1e606:	mov    edx,DWORD PTR [rbx]
   1e608:	cmp    esi,edx
   1e60a:	jne    1f650 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1280>
   1e610:	mov    edi,DWORD PTR [rbp+0x0]
   1e613:	test   edi,edi
   1e615:	js     1e760 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x390>
   1e61b:	mov    rax,QWORD PTR [rbp+0x108]
   1e622:	cmp    rbx,rax
   1e625:	je     1e770 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x3a0>
   1e62b:	cmp    esi,DWORD PTR [rbx]
   1e62d:	jne    1f6ca <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x12fa>
   1e633:	and    ecx,0xeffffff8
   1e639:	mov    DWORD PTR [rbx],ecx
   1e63b:	pause
   1e63d:	mov    eax,DWORD PTR [rbp+0x0]
   1e640:	test   eax,eax
   1e642:	js     1e660 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x290>
   1e644:	mov    rbx,QWORD PTR [rbp+0x108]
   1e64b:	test   rbx,rbx
   1e64e:	je     1e66c <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x29c>
   1e650:	mov    ecx,DWORD PTR [rbx]
   1e652:	test   cl,0x1
   1e655:	je     1e5f0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x220>
   1e657:	pause
   1e659:	mov    ecx,DWORD PTR [rbx]
   1e65b:	jmp    1e652 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x282>
   1e65d:	nop    DWORD PTR [rax]
   1e660:	mov    rbx,QWORD PTR [rbp+0x128]
   1e667:	test   rbx,rbx
   1e66a:	jne    1e650 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x280>
   1e66c:	mov    r12d,0xffffffff
   1e672:	mov    rax,QWORD PTR [rsp+0x10]
   1e677:	mov    rax,QWORD PTR [rax+0x60]
   1e67b:	test   rax,rax
   1e67e:	je     1ea2c <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x65c>
   1e684:	mov    rdx,QWORD PTR [rax]
   1e687:	mov    rdi,QWORD PTR [rsp+0x10]
   1e68c:	mov    QWORD PTR [rdi+0x60],rdx
   1e690:	mov    edi,DWORD PTR [rsp+0x18]
   1e694:	mov    edx,DWORD PTR [rsp+0x8]
   1e698:	mov    BYTE PTR [rax+0x4],0x0
   1e69c:	mov    DWORD PTR [rax],0x0
   1e6a2:	mov    ecx,DWORD PTR [r15]
   1e6a5:	mov    DWORD PTR [rax+0x8],edi
   1e6a8:	mov    rdx,QWORD PTR [rsp+rdx*8+0x70]
   1e6ad:	mov    QWORD PTR [rax+0x108],0x0
   1e6b8:	mov    QWORD PTR [rax+0x88],rbp
   1e6bf:	test   ecx,ecx
   1e6c1:	js     1ea20 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x650>
   1e6c7:	mov    QWORD PTR [r15+0x108],rax
   1e6ce:	mov    QWORD PTR [rax+0x90],r15
   1e6d5:	mov    QWORD PTR [rax+0x10],rdx
   1e6d9:	mov    BYTE PTR [rax+0x4],0x1
   1e6dd:	test   r12d,r12d
   1e6e0:	js     1ea00 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x630>
   1e6e6:	movsxd r12,r12d
   1e6e9:	mov    QWORD PTR [rax+0x108],rbx
   1e6f0:	mov    QWORD PTR [rbx+r12*8+0x88],rax
   1e6f8:	mov    edx,DWORD PTR [rbp+0x0]
   1e6fb:	test   edx,edx
   1e6fd:	js     1e9f0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x620>
   1e703:	mov    QWORD PTR [rbp+0x108],rax
   1e70a:	xor    r13d,r13d
   1e70d:	cmp    QWORD PTR [r14],rbp
   1e710:	je     1e8d0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x500>
   1e716:	mov    esi,DWORD PTR [rbp+0x0]
   1e719:	mov    rdi,rbp
   1e71c:	call   174e0 <_ZN11nodeversionI22nodeversion_parametersIjEE6unlockES2_>
   1e721:	mov    rax,QWORD PTR [r14]
   1e724:	cmp    r15,rax
   1e727:	je     1e734 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x364>
   1e729:	mov    esi,DWORD PTR [r15]
   1e72c:	mov    rdi,r15
   1e72f:	call   174e0 <_ZN11nodeversionI22nodeversion_parametersIjEE6unlockES2_>
   1e734:	test   r13,r13
   1e737:	je     1ea9b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x6cb>
   1e73d:	mov    eax,DWORD PTR [rbx]
   1e73f:	xor    DWORD PTR [rsp+0x8],0x1
   1e744:	add    DWORD PTR [rsp+0x18],0x1
   1e749:	test   al,0x1
   1e74b:	je     1f720 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1350>
   1e751:	mov    rbp,rbx
   1e754:	mov    r15,r13
   1e757:	jmp    1e5d0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x200>
   1e75c:	nop    DWORD PTR [rax+0x0]
   1e760:	mov    rax,QWORD PTR [rbp+0x128]
   1e767:	cmp    rbx,rax
   1e76a:	jne    1e62b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x25b>
   1e770:	test   edx,edx
   1e772:	js     1f75e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x138e>
   1e778:	movzx  ecx,BYTE PTR [rbx+0x4]
   1e77c:	mov    edi,DWORD PTR [rsp+0x8]
   1e780:	xor    r12d,r12d
   1e783:	cmp    ecx,r12d
   1e786:	jle    1e7b0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x3e0>
   1e788:	lea    eax,[rcx+r12*1]
   1e78c:	sar    eax,1
   1e78e:	movsxd rsi,eax
   1e791:	mov    rsi,QWORD PTR [rbx+rsi*8+0x10]
   1e796:	cmp    QWORD PTR [rsp+rdi*8+0x70],rsi
   1e79b:	je     1e9d0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x600>
   1e7a1:	jb     1e9e0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x610>
   1e7a7:	lea    r12d,[rax+0x1]
   1e7ab:	cmp    ecx,r12d
   1e7ae:	jg     1e788 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x3b8>
   1e7b0:	test   dl,0x1
   1e7b3:	je     1f820 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1450>
   1e7b9:	or     edx,0x2
   1e7bc:	mov    DWORD PTR [rbx],edx
   1e7be:	test   r12d,r12d
   1e7c1:	js     1e672 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x2a2>
   1e7c7:	mov    eax,DWORD PTR [rsp+0x18]
   1e7cb:	cmp    eax,DWORD PTR [rbx+0x8]
   1e7ce:	jb     1e672 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x2a2>
   1e7d4:	movzx  eax,BYTE PTR [rbx+0x4]
   1e7d8:	mov    r9d,DWORD PTR [rsp+0x8]
   1e7dd:	xor    r13d,r13d
   1e7e0:	cmp    al,0xe
   1e7e2:	ja     1ead5 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x705>
   1e7e8:	movzx  ecx,al
   1e7eb:	lea    eax,[r12+0x1]
   1e7f0:	lea    rdi,[rbx+0x10]
   1e7f4:	mov    QWORD PTR [rsp+0x38],r9
   1e7f9:	sub    ecx,r12d
   1e7fc:	movsxd r12,r12d
   1e7ff:	mov    DWORD PTR [rsp+0x20],eax
   1e803:	movsxd r11,ecx
   1e806:	lea    rax,[r12*8+0x0]
   1e80e:	mov    DWORD PTR [rsp+0x30],ecx
   1e812:	lea    rdx,[r11*8+0x0]
   1e81a:	lea    rsi,[rdi+rax*1]
   1e81e:	mov    QWORD PTR [rsp+0x28],r11
   1e823:	lea    rdi,[rdi+rax*1+0x8]
   1e828:	call   5d40 <memmove@plt>
   1e82d:	mov    r11,QWORD PTR [rsp+0x28]
   1e832:	movsxd r8,DWORD PTR [rsp+0x20]
   1e837:	lea    rdi,[rbx+0x88]
   1e83e:	mov    ecx,DWORD PTR [rsp+0x30]
   1e842:	mov    r9,QWORD PTR [rsp+0x38]
   1e847:	lea    rdx,[r8+r11*1]
   1e84b:	lea    rax,[r11+r12*1]
   1e84f:	shl    rdx,0x3
   1e853:	shl    rax,0x3
   1e857:	test   ecx,ecx
   1e859:	je     1e899 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x4c9>
   1e85b:	lea    esi,[rcx-0x1]
   1e85e:	mov    QWORD PTR [rsp+0x28],r9
   1e863:	mov    rcx,rsi
   1e866:	lea    r11,[rsi*8+0x8]
   1e86e:	mov    QWORD PTR [rsp+0x20],r8
   1e873:	neg    rcx
   1e876:	shl    rcx,0x3
   1e87a:	add    rdx,rcx
   1e87d:	add    rax,rcx
   1e880:	lea    rsi,[rdi+rax*1]
   1e884:	add    rdi,rdx
   1e887:	mov    rdx,r11
   1e88a:	call   5d40 <memmove@plt>
   1e88f:	mov    r8,QWORD PTR [rsp+0x20]
   1e894:	mov    r9,QWORD PTR [rsp+0x28]
   1e899:	mov    ecx,DWORD PTR [r15]
   1e89c:	mov    rax,QWORD PTR [rsp+r9*8+0x70]
   1e8a1:	test   ecx,ecx
   1e8a3:	js     1f192 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xdc2>
   1e8a9:	mov    QWORD PTR [r15+0x108],rbx
   1e8b0:	mov    QWORD PTR [rbx+r8*8+0x88],r15
   1e8b8:	mov    QWORD PTR [rbx+r12*8+0x10],rax
   1e8bd:	add    BYTE PTR [rbx+0x4],0x1
   1e8c1:	cmp    QWORD PTR [r14],rbp
   1e8c4:	jne    1e716 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x346>
   1e8ca:	nop    WORD PTR [rax+rax*1+0x0]
   1e8d0:	mov    rax,QWORD PTR [rbp+0x18]
   1e8d4:	mov    ecx,DWORD PTR [r15+0x18]
   1e8d8:	mov    edx,eax
   1e8da:	and    ecx,0xf
   1e8dd:	and    rax,0xfffffffffffffff0
   1e8e1:	and    edx,0xf
   1e8e4:	mov    r12d,edx
   1e8e7:	sub    r12d,ecx
   1e8ea:	movsxd r12,r12d
   1e8ed:	or     r12,rax
   1e8f0:	cmp    edx,0xf
   1e8f3:	je     1e925 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x555>
   1e8f5:	lea    ecx,[r12*4+0x0]
   1e8fd:	mov    rax,r12
   1e900:	mov    rdx,r12
   1e903:	and    ecx,0x3c
   1e906:	shr    rdx,0x38
   1e90a:	shr    rax,cl
   1e90d:	xor    rax,rdx
   1e910:	and    eax,0xf0
   1e915:	mov    rdx,rax
   1e918:	shl    rax,0x38
   1e91c:	shl    rdx,cl
   1e91f:	or     rax,rdx
   1e922:	xor    r12,rax
   1e925:	mov    eax,DWORD PTR [rbp+0x0]
   1e928:	test   al,0x1
   1e92a:	je     1f84d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x147d>
   1e930:	or     eax,0x4
   1e933:	mov    DWORD PTR [rbp+0x0],eax
   1e936:	mov    edx,DWORD PTR [rsp+0x1c]
   1e93a:	mov    QWORD PTR [rbp+0x18],r12
   1e93e:	test   edx,edx
   1e940:	jne    1ea50 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x680>
   1e946:	shr    r12,0x3c
   1e94a:	mov    DWORD PTR [r14+0x2c],r12d
   1e94e:	mov    r8d,r12d
   1e951:	movsxd rdx,r12d
   1e954:	mov    QWORD PTR [rbp+r12*8+0x98],0x0
   1e960:	mov    rax,QWORD PTR [r14+0x8]
   1e964:	mov    QWORD PTR [rbp+r12*8+0x20],rax
   1e969:	mov    eax,DWORD PTR [r14+0x10]
   1e96d:	cmp    eax,0x8
   1e970:	jg     1ee2c <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa5c>
   1e976:	mov    BYTE PTR [rbp+rdx*1+0x6],al
   1e97a:	mov    esi,DWORD PTR [r15]
   1e97d:	mov    DWORD PTR [rsp+0x6c],esi
   1e981:	test   sil,0x1
   1e985:	jne    1f070 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xca0>
   1e98b:	mov    r12d,DWORD PTR [r15+0x18]
   1e98f:	shl    esi,0x4
   1e992:	mov    rax,QWORD PTR [r14+0x58]
   1e996:	and    r12d,0xf
   1e99a:	add    r12d,esi
   1e99d:	cmp    rax,QWORD PTR [r14+0x60]
   1e9a1:	je     1f19e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xdce>
   1e9a7:	mov    QWORD PTR [rax],r15
   1e9aa:	add    rax,0x10
   1e9ae:	mov    DWORD PTR [rax-0x8],r12d
   1e9b2:	mov    QWORD PTR [r14+0x58],rax
   1e9b6:	mov    rax,QWORD PTR [r14]
   1e9b9:	cmp    rbp,rax
   1e9bc:	je     1e724 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x354>
   1e9c2:	jmp    1e716 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x346>
   1e9c7:	nop    WORD PTR [rax+rax*1+0x0]
   1e9d0:	lea    r12d,[rax+0x1]
   1e9d4:	jmp    1e7b0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x3e0>
   1e9d9:	nop    DWORD PTR [rax+0x0]
   1e9e0:	mov    ecx,eax
   1e9e2:	jmp    1e783 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x3b3>
   1e9e7:	nop    WORD PTR [rax+rax*1+0x0]
   1e9f0:	mov    QWORD PTR [rbp+0x128],rax
   1e9f7:	jmp    1e70a <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x33a>
   1e9fc:	nop    DWORD PTR [rax+0x0]
   1ea00:	mov    QWORD PTR [rax+0x108],0x0
   1ea0b:	mov    DWORD PTR [rax],0x40000000
   1ea11:	jmp    1e6f8 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x328>
   1ea16:	cs nop WORD PTR [rax+rax*1+0x0]
   1ea20:	mov    QWORD PTR [r15+0x128],rax
   1ea27:	jmp    1e6ce <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x2fe>
   1ea2c:	mov    r13,QWORD PTR [rsp+0x10]
   1ea31:	mov    esi,0x5
   1ea36:	mov    rdi,r13
   1ea39:	call   59690 <_ZN10threadinfo11refill_poolEi>
   1ea3e:	mov    rax,QWORD PTR [r13+0x60]
   1ea42:	test   rax,rax
   1ea45:	je     1e690 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x2c0>
   1ea4b:	jmp    1e684 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x2b4>
   1ea50:	mov    eax,DWORD PTR [r14+0x28]
   1ea54:	and    r12d,0xf
   1ea58:	mov    QWORD PTR [r14],r15
   1ea5b:	sub    eax,r12d
   1ea5e:	movd   xmm0,eax
   1ea62:	pshufd xmm0,xmm0,0xe0
   1ea67:	movq   QWORD PTR [r14+0x28],xmm0
   1ea6d:	mov    esi,DWORD PTR [rbp+0x0]
   1ea70:	mov    DWORD PTR [rsp+0x6c],esi
   1ea74:	test   sil,0x1
   1ea78:	jne    1f083 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xcb3>
   1ea7e:	shl    esi,0x4
   1ea81:	lea    eax,[rsi+r12*1]
   1ea85:	mov    DWORD PTR [r14+0x4c],eax
   1ea89:	cmp    r15,rbp
   1ea8c:	jne    1e716 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x346>
   1ea92:	test   r13,r13
   1ea95:	jne    1e73d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x36d>
   1ea9b:	test   rbx,rbx
   1ea9e:	je     1eaaa <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x6da>
   1eaa0:	mov    esi,DWORD PTR [rbx]
   1eaa2:	mov    rdi,rbx
   1eaa5:	call   174e0 <_ZN11nodeversionI22nodeversion_parametersIjEE6unlockES2_>
   1eaaa:	mov    rax,QWORD PTR [rsp+0x88]
   1eab2:	sub    rax,QWORD PTR fs:0x28
   1eabb:	jne    1f816 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1446>
   1eac1:	add    rsp,0x98
   1eac8:	xor    eax,eax
   1eaca:	pop    rbx
   1eacb:	pop    rbp
   1eacc:	pop    r12
   1eace:	pop    r13
   1ead0:	pop    r14
   1ead2:	pop    r15
   1ead4:	ret
   1ead5:	mov    rax,QWORD PTR [rsp+0x10]
   1eada:	mov    r13,QWORD PTR [rax+0x60]
   1eade:	test   r13,r13
   1eae1:	je     1f211 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xe41>
   1eae7:	mov    rax,QWORD PTR [r13+0x0]
   1eaeb:	mov    rdi,QWORD PTR [rsp+0x10]
   1eaf0:	mov    QWORD PTR [rdi+0x60],rax
   1eaf4:	mov    eax,DWORD PTR [rsp+0x18]
   1eaf8:	mov    BYTE PTR [r13+0x4],0x0
   1eafd:	mov    DWORD PTR [r13+0x0],0x0
   1eb05:	mov    DWORD PTR [r13+0x8],eax
   1eb09:	mov    eax,DWORD PTR [rbx]
   1eb0b:	mov    QWORD PTR [r13+0x108],0x0
   1eb16:	mov    DWORD PTR [r13+0x0],eax
   1eb1a:	test   al,0x1
   1eb1c:	je     1f78f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x13bf>
   1eb22:	and    eax,0xbfffffff
   1eb27:	mov    DWORD PTR [r13+0x0],eax
   1eb2b:	mov    r9d,DWORD PTR [rsp+0x8]
   1eb30:	mov    r11d,r9d
   1eb33:	mov    rcx,QWORD PTR [rsp+r9*8+0x70]
   1eb38:	xor    r11d,0x1
   1eb3c:	test   BYTE PTR [rbx],0x1
   1eb3f:	je     1f7bc <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x13ec>
   1eb45:	test   BYTE PTR [r13+0x0],0x1
   1eb4a:	je     1f7bc <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x13ec>
   1eb50:	movzx  eax,BYTE PTR [rsp+0x55]
   1eb55:	mov    BYTE PTR [r13+0x4],al
   1eb59:	movsxd rax,DWORD PTR [rsp+0x50]
   1eb5e:	cmp    r12d,eax
   1eb61:	jl     1f1db <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xe0b>
   1eb67:	je     1f30f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xf3f>
   1eb6d:	add    eax,0x1
   1eb70:	mov    r10d,r12d
   1eb73:	movsxd rsi,eax
   1eb76:	sub    r10d,eax
   1eb79:	mov    rdx,QWORD PTR [rbx+rsi*8+0x88]
   1eb81:	mov    QWORD PTR [r13+0x88],rdx
   1eb88:	cmp    rbx,r13
   1eb8b:	je     1f7e9 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1419>
   1eb91:	xor    r8d,r8d
   1eb94:	test   r10d,r10d
   1eb97:	jne    1f4fc <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x112c>
   1eb9d:	mov    eax,DWORD PTR [r15]
   1eba0:	test   eax,eax
   1eba2:	js     1f3e8 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1018>
   1eba8:	mov    QWORD PTR [r15+0x108],r13
   1ebaf:	lea    eax,[r10+0x1]
   1ebb3:	cdqe
   1ebb5:	mov    QWORD PTR [r13+rax*8+0x88],r15
   1ebbd:	mov    eax,0xf
   1ebc2:	sub    eax,r12d
   1ebc5:	mov    QWORD PTR [r13+r8*8+0x10],rcx
   1ebca:	movsxd rdx,eax
   1ebcd:	jne    1f47f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x10af>
   1ebd3:	movsxd rax,DWORD PTR [rsp+0x50]
   1ebd8:	mov    rcx,QWORD PTR [rbx+rax*8+0x10]
   1ebdd:	mov    rax,QWORD PTR [r13+0x88]
   1ebe4:	mov    QWORD PTR [rsp+r11*8+0x70],rcx
   1ebe9:	mov    edi,DWORD PTR [rax]
   1ebeb:	test   edi,edi
   1ebed:	js     1f1cf <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xdff>
   1ebf3:	mov    QWORD PTR [rax+0x108],r13
   1ebfa:	cmp    BYTE PTR [r13+0x4],0x0
   1ebff:	je     1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ec05:	mov    rax,QWORD PTR [r13+0x90]
   1ec0c:	mov    esi,DWORD PTR [rax]
   1ec0e:	test   esi,esi
   1ec10:	js     1f238 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xe68>
   1ec16:	mov    QWORD PTR [rax+0x108],r13
   1ec1d:	cmp    BYTE PTR [r13+0x4],0x1
   1ec22:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ec28:	mov    rax,QWORD PTR [r13+0x98]
   1ec2f:	mov    ecx,DWORD PTR [rax]
   1ec31:	test   ecx,ecx
   1ec33:	js     1f244 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xe74>
   1ec39:	mov    QWORD PTR [rax+0x108],r13
   1ec40:	cmp    BYTE PTR [r13+0x4],0x2
   1ec45:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ec4b:	mov    rax,QWORD PTR [r13+0xa0]
   1ec52:	mov    edx,DWORD PTR [rax]
   1ec54:	test   edx,edx
   1ec56:	js     1f250 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xe80>
   1ec5c:	mov    QWORD PTR [rax+0x108],r13
   1ec63:	cmp    BYTE PTR [r13+0x4],0x3
   1ec68:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ec6e:	mov    rax,QWORD PTR [r13+0xa8]
   1ec75:	mov    r11d,DWORD PTR [rax]
   1ec78:	test   r11d,r11d
   1ec7b:	js     1f303 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xf33>
   1ec81:	mov    QWORD PTR [rax+0x108],r13
   1ec88:	cmp    BYTE PTR [r13+0x4],0x4
   1ec8d:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ec93:	mov    rax,QWORD PTR [r13+0xb0]
   1ec9a:	mov    r10d,DWORD PTR [rax]
   1ec9d:	test   r10d,r10d
   1eca0:	js     1f3d0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1000>
   1eca6:	mov    QWORD PTR [rax+0x108],r13
   1ecad:	cmp    BYTE PTR [r13+0x4],0x5
   1ecb2:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ecb8:	mov    rax,QWORD PTR [r13+0xb8]
   1ecbf:	mov    r8d,DWORD PTR [rax]
   1ecc2:	test   r8d,r8d
   1ecc5:	js     1f3dc <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x100c>
   1eccb:	mov    QWORD PTR [rax+0x108],r13
   1ecd2:	cmp    BYTE PTR [r13+0x4],0x6
   1ecd7:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ecdd:	mov    rax,QWORD PTR [r13+0xc0]
   1ece4:	mov    edi,DWORD PTR [rax]
   1ece6:	test   edi,edi
   1ece8:	js     1f433 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1063>
   1ecee:	mov    QWORD PTR [rax+0x108],r13
   1ecf5:	cmp    BYTE PTR [r13+0x4],0x7
   1ecfa:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ed00:	mov    rax,QWORD PTR [r13+0xc8]
   1ed07:	mov    esi,DWORD PTR [rax]
   1ed09:	test   esi,esi
   1ed0b:	js     1f43f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x106f>
   1ed11:	mov    QWORD PTR [rax+0x108],r13
   1ed18:	cmp    BYTE PTR [r13+0x4],0x8
   1ed1d:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ed23:	mov    rax,QWORD PTR [r13+0xd0]
   1ed2a:	mov    ecx,DWORD PTR [rax]
   1ed2c:	test   ecx,ecx
   1ed2e:	js     1f5f4 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1224>
   1ed34:	mov    QWORD PTR [rax+0x108],r13
   1ed3b:	cmp    BYTE PTR [r13+0x4],0x9
   1ed40:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ed46:	mov    rax,QWORD PTR [r13+0xd8]
   1ed4d:	mov    edx,DWORD PTR [rax]
   1ed4f:	test   edx,edx
   1ed51:	js     1f5e8 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1218>
   1ed57:	mov    QWORD PTR [rax+0x108],r13
   1ed5e:	cmp    BYTE PTR [r13+0x4],0xa
   1ed63:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ed69:	mov    rax,QWORD PTR [r13+0xe0]
   1ed70:	mov    r11d,DWORD PTR [rax]
   1ed73:	test   r11d,r11d
   1ed76:	js     1f407 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1037>
   1ed7c:	mov    QWORD PTR [rax+0x108],r13
   1ed83:	cmp    BYTE PTR [r13+0x4],0xb
   1ed88:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1ed8a:	mov    rax,QWORD PTR [r13+0xe8]
   1ed91:	mov    r10d,DWORD PTR [rax]
   1ed94:	test   r10d,r10d
   1ed97:	js     1f644 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1274>
   1ed9d:	mov    QWORD PTR [rax+0x108],r13
   1eda4:	cmp    BYTE PTR [r13+0x4],0xc
   1eda9:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1edab:	mov    rax,QWORD PTR [r13+0xf0]
   1edb2:	mov    r8d,DWORD PTR [rax]
   1edb5:	test   r8d,r8d
   1edb8:	js     1f638 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1268>
   1edbe:	mov    QWORD PTR [rax+0x108],r13
   1edc5:	cmp    BYTE PTR [r13+0x4],0xd
   1edca:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1edcc:	mov    rax,QWORD PTR [r13+0xf8]
   1edd3:	mov    edi,DWORD PTR [rax]
   1edd5:	test   edi,edi
   1edd7:	js     1f6be <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x12ee>
   1eddd:	mov    QWORD PTR [rax+0x108],r13
   1ede4:	cmp    BYTE PTR [r13+0x4],0xe
   1ede9:	jbe    1ee03 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xa33>
   1edeb:	mov    rax,QWORD PTR [r13+0x100]
   1edf2:	mov    esi,DWORD PTR [rax]
   1edf4:	test   esi,esi
   1edf6:	js     1f6b2 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x12e2>
   1edfc:	mov    QWORD PTR [rax+0x108],r13
   1ee03:	mov    eax,DWORD PTR [rbx]
   1ee05:	test   al,0x1
   1ee07:	je     1f84d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x147d>
   1ee0d:	or     eax,0x4
   1ee10:	mov    DWORD PTR [rbx],eax
   1ee12:	mov    eax,DWORD PTR [rsp+0x50]
   1ee16:	cmp    r12d,eax
   1ee19:	jge    1f751 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1381>
   1ee1f:	movzx  eax,BYTE PTR [rsp+0x56]
   1ee24:	mov    BYTE PTR [rbx+0x4],al
   1ee27:	jmp    1e7e8 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x418>
   1ee2c:	mov    BYTE PTR [rbp+rdx*1+0x6],0x40
   1ee31:	mov    eax,DWORD PTR [r14+0x10]
   1ee35:	mov    rdi,QWORD PTR [rbp+0x110]
   1ee3c:	lea    r9d,[rax-0x8]
   1ee40:	mov    rax,QWORD PTR [r14+0x18]
   1ee44:	add    rax,0x8
   1ee48:	mov    QWORD PTR [rsp+0x28],rax
   1ee4d:	test   rdi,rdi
   1ee50:	je     1ee7c <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xaac>
   1ee52:	mov    ecx,r9d
   1ee55:	mov    rdx,rax
   1ee58:	mov    esi,r12d
   1ee5b:	mov    DWORD PTR [rsp+0x30],r9d
   1ee60:	mov    DWORD PTR [rsp+0x20],r12d
   1ee65:	call   1d140 <_ZN9stringbagItE6assignEiPKci>
   1ee6a:	mov    r8d,DWORD PTR [rsp+0x20]
   1ee6f:	mov    r9d,DWORD PTR [rsp+0x30]
   1ee74:	test   al,al
   1ee76:	jne    1e97a <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x5aa>
   1ee7c:	cmp    BYTE PTR [rbp+0x4],0x0
   1ee80:	mov    eax,r9d
   1ee83:	jg     1f25c <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xe8c>
   1ee89:	mov    r11,QWORD PTR [rbp+0x18]
   1ee8d:	mov    rdi,QWORD PTR [rbp+0x110]
   1ee94:	mov    esi,r11d
   1ee97:	mov    QWORD PTR [rsp+0x20],rdi
   1ee9c:	and    esi,0xf
   1ee9f:	je     1eeef <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xb1f>
   1eea1:	lea    edx,[rsi*4+0x4]
   1eea8:	test   rdi,rdi
   1eeab:	je     1f44b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x107b>
   1eeb1:	xor    edi,edi
   1eeb3:	mov    ecx,0x4
   1eeb8:	jmp    1eec1 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xaf1>
   1eeba:	add    ecx,0x4
   1eebd:	cmp    edx,ecx
   1eebf:	je     1eeeb <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xb1b>
   1eec1:	mov    rax,r11
   1eec4:	shr    rax,cl
   1eec7:	and    eax,0xf
   1eeca:	cmp    r8d,eax
   1eecd:	je     1eeba <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xaea>
   1eecf:	cmp    BYTE PTR [rbp+rax*1+0x6],0x40
   1eed4:	jne    1eeba <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xaea>
   1eed6:	mov    r10,QWORD PTR [rsp+0x20]
   1eedb:	add    ecx,0x4
   1eede:	movzx  eax,WORD PTR [r10+rax*4+0x6]
   1eee4:	add    rdi,rax
   1eee7:	cmp    edx,ecx
   1eee9:	jne    1eec1 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xaf1>
   1eeeb:	lea    eax,[r9+rdi*1]
   1eeef:	lea    rdx,[rax+0x47]
   1eef3:	add    rax,0x46
   1eef7:	mov    r12d,0x1
   1eefd:	and    rax,rdx
   1ef00:	bsr    rcx,rdx
   1ef04:	cmp    rax,0x1
   1ef08:	mov    rax,QWORD PTR [rsp+0x20]
   1ef0d:	sbb    ecx,0xffffffff
   1ef10:	shl    r12,cl
   1ef13:	test   rax,rax
   1ef16:	je     1ef27 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xb57>
   1ef18:	movzx  eax,WORD PTR [rax+0x2]
   1ef1c:	add    rax,0x1
   1ef20:	cmp    r12,rax
   1ef23:	cmovb  r12,rax
   1ef27:	mov    rdi,r12
   1ef2a:	mov    DWORD PTR [rsp+0x48],esi
   1ef2e:	mov    QWORD PTR [rsp+0x40],r11
   1ef33:	mov    DWORD PTR [rsp+0x38],r9d
   1ef38:	mov    DWORD PTR [rsp+0x30],r8d
   1ef3d:	call   5ba0 <malloc@plt>
   1ef42:	cmp    r12,0x10000
   1ef49:	mov    r8d,DWORD PTR [rsp+0x30]
   1ef4e:	mov    r9d,DWORD PTR [rsp+0x38]
   1ef53:	mov    r11,QWORD PTR [rsp+0x40]
   1ef58:	mov    esi,DWORD PTR [rsp+0x48]
   1ef5c:	mov    rdi,rax
   1ef5f:	ja     1f8f7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0x1527>
   1ef65:	sub    r12d,0x1
   1ef69:	mov    eax,0x40
   1ef6e:	mov    DWORD PTR [rdi+0x4],0x0
   1ef75:	lea    r10d,[rsi*4+0x4]
   1ef7d:	mov    WORD PTR [rdi+0x2],r12w
   1ef82:	mov    r12d,0x4
   1ef88:	mov    WORD PTR [rdi],ax
   1ef8b:	mov    QWORD PTR [rdi+0x8],0x0
   1ef93:	mov    QWORD PTR [rdi+0x10],0x0
   1ef9b:	mov    QWORD PTR [rdi+0x18],0x0
   1efa3:	mov    QWORD PTR [rdi+0x20],0x0
   1efab:	mov    QWORD PTR [rdi+0x28],0x0
   1efb3:	mov    QWORD PTR [rdi+0x30],0x0
   1efbb:	mov    QWORD PTR [rdi+0x38],0x0
   1efc3:	test   esi,esi
   1efc5:	je     1f0c7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xcf7>
   1efcb:	mov    QWORD PTR [rsp+0x30],r13
   1efd0:	mov    r13d,r8d
   1efd3:	mov    QWORD PTR [rsp+0x38],r15
   1efd8:	mov    r15,r11
   1efdb:	mov    QWORD PTR [rsp+0x40],rbx
   1efe0:	mov    rbx,rdi
   1efe3:	mov    QWORD PTR [rsp+0x58],r14
   1efe8:	mov    r14d,r10d
   1efeb:	mov    DWORD PTR [rsp+0x48],r9d
   1eff0:	jmp    1f005 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE10make_splitER10threadinfo+0xc35>
   1eff2:	nop    WORD PTR [rax+rax*1+0x0]
   1eff8:	add    r12d,0x4
   1effc:	cmp    r12d,r14d
   1efff:	.byte 0xf
