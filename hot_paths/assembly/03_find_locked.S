mttest:     file format elf64-x86-64


Disassembly of section .text:

000000000001b190 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo>:
   1b190:	endbr64
   1b194:	push   r15
   1b196:	push   r14
   1b198:	push   r13
   1b19a:	mov    r13,rdi
   1b19d:	push   r12
   1b19f:	push   rbp
   1b1a0:	lea    rbp,[rdi+0x8]
   1b1a4:	push   rbx
   1b1a5:	sub    rsp,0x28
   1b1a9:	mov    rbx,QWORD PTR [rdi+0x30]
   1b1ad:	mov    rax,QWORD PTR fs:0x28
   1b1b6:	mov    QWORD PTR [rsp+0x18],rax
   1b1bb:	xor    eax,eax
   1b1bd:	lea    r12,[rsp+0xc]
   1b1c2:	mov    rdx,r12
   1b1c5:	mov    rsi,rbp
   1b1c8:	mov    rdi,rbx
   1b1cb:	call   afb0 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0>
   1b1d0:	mov    QWORD PTR [r13+0x0],rax
   1b1d4:	mov    rdx,rax
   1b1d7:	test   BYTE PTR [rsp+0xf],0x20
   1b1dc:	jne    1b1c2 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x32>
   1b1de:	xchg   ax,ax
   1b1e0:	prefetcht0 BYTE PTR [rdx+0x40]
   1b1e4:	prefetcht0 BYTE PTR [rdx+0x80]
   1b1eb:	prefetcht0 BYTE PTR [rdx+0xc0]
   1b1f2:	cmp    BYTE PTR [rdx+0x4],0x0
   1b1f6:	jle    1b530 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x3a0>
   1b1fc:	prefetcht0 BYTE PTR [rdx+0x138]
   1b203:	mov    r15,QWORD PTR [rdx+0x18]
   1b207:	mov    r14,QWORD PTR [r13+0x0]
   1b20b:	mov    r9,QWORD PTR [r13+0x8]
   1b20f:	xor    esi,esi
   1b211:	mov    r10d,DWORD PTR [r13+0x10]
   1b215:	mov    r8,QWORD PTR [r14+0x18]
   1b219:	mov    edi,r8d
   1b21c:	and    edi,0xf
   1b21f:	je     1b262 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xd2>
   1b221:	cmp    r10d,0x8
   1b225:	jg     1b344 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1b4>
   1b22b:	nop    DWORD PTR [rax+rax*1+0x0]
   1b230:	lea    edx,[rdi+rsi*1]
   1b233:	mov    rax,r8
   1b236:	sar    edx,1
   1b238:	lea    ecx,[rdx*4+0x4]
   1b23f:	shr    rax,cl
   1b242:	mov    ecx,eax
   1b244:	and    eax,0xf
   1b247:	and    ecx,0xf
   1b24a:	cmp    r9,QWORD PTR [r14+rax*8+0x20]
   1b24f:	je     1b448 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x2b8>
   1b255:	jb     1b380 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1f0>
   1b25b:	lea    esi,[rdx+0x1]
   1b25e:	cmp    edi,esi
   1b260:	jg     1b230 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xa0>
   1b262:	mov    DWORD PTR [r13+0x28],esi
   1b266:	xor    eax,eax
   1b268:	mov    DWORD PTR [r13+0x2c],0xffffffff
   1b270:	mov    DWORD PTR [r13+0x38],eax
   1b274:	mov    edx,DWORD PTR [rsp+0xc]
   1b278:	test   dl,0x1
   1b27b:	jne    1b500 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x370>
   1b281:	mov    ecx,edx
   1b283:	mov    eax,edx
   1b285:	or     ecx,0x1
   1b288:	lock cmpxchg DWORD PTR [r14],ecx
   1b28d:	jne    1b500 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x370>
   1b293:	and    edx,0x6
   1b296:	jne    1b855 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x6c5>
   1b29c:	cmp    ecx,DWORD PTR [r14]
   1b29f:	jne    1b82b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x69b>
   1b2a5:	mov    rdx,QWORD PTR [r13+0x0]
   1b2a9:	mov    eax,DWORD PTR [rsp+0xc]
   1b2ad:	xor    eax,DWORD PTR [rdx]
   1b2af:	mov    r14,QWORD PTR [r13+0x0]
   1b2b3:	cmp    eax,0x1
   1b2b6:	ja     1b2c2 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x132>
   1b2b8:	cmp    r15,QWORD PTR [r14+0x18]
   1b2bc:	je     1b68b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x4fb>
   1b2c2:	mov    edx,DWORD PTR [r14]
   1b2c5:	cmp    edx,DWORD PTR [r14]
   1b2c8:	jne    1b801 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x671>
   1b2ce:	test   dl,0x1
   1b2d1:	je     1b7d3 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x643>
   1b2d7:	test   dl,0x4
   1b2da:	je     1b510 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x380>
   1b2e0:	lea    eax,[rdx+0x200]
   1b2e6:	and    eax,0xaffffe00
   1b2eb:	mov    rdx,QWORD PTR [r13+0x0]
   1b2ef:	mov    DWORD PTR [r14],eax
   1b2f2:	mov    ecx,DWORD PTR [rsp+0xc]
   1b2f6:	mov    eax,DWORD PTR [rdx]
   1b2f8:	test   al,0x6
   1b2fa:	je     1b308 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x178>
   1b2fc:	nop    DWORD PTR [rax+0x0]
   1b300:	pause
   1b302:	mov    eax,DWORD PTR [rdx]
   1b304:	test   al,0x6
   1b306:	jne    1b300 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x170>
   1b308:	mov    DWORD PTR [rsp+0xc],eax
   1b30c:	mov    esi,DWORD PTR [rsp+0xc]
   1b310:	mov    eax,ecx
   1b312:	xor    eax,esi
   1b314:	mov    ecx,esi
   1b316:	and    ecx,0x20000000
   1b31c:	cmp    eax,0x1ff
   1b321:	ja     1b3a0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x210>
   1b323:	mov    QWORD PTR [r13+0x0],rdx
   1b327:	test   ecx,ecx
   1b329:	je     1b1e0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x50>
   1b32f:	jmp    1b1c2 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x32>
   1b334:	nop    DWORD PTR [rax+0x0]
   1b338:	jae    1b371 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1e1>
   1b33a:	mov    edi,edx
   1b33c:	cmp    esi,edi
   1b33e:	jge    1b262 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xd2>
   1b344:	lea    edx,[rsi+rdi*1]
   1b347:	mov    rax,r8
   1b34a:	sar    edx,1
   1b34c:	lea    ecx,[rdx*4+0x4]
   1b353:	shr    rax,cl
   1b356:	mov    ecx,eax
   1b358:	and    eax,0xf
   1b35b:	and    ecx,0xf
   1b35e:	cmp    r9,QWORD PTR [r14+rax*8+0x20]
   1b363:	jne    1b338 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1a8>
   1b365:	cmp    BYTE PTR [r14+rax*1+0x6],0x8
   1b36b:	ja     1b45d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x2cd>
   1b371:	lea    esi,[rdx+0x1]
   1b374:	jmp    1b33c <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1ac>
   1b376:	cs nop WORD PTR [rax+rax*1+0x0]
   1b380:	mov    edi,edx
   1b382:	jmp    1b25e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xce>
   1b387:	mov    esi,ecx
   1b389:	test   cl,0x6
   1b38c:	je     1b398 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x208>
   1b38e:	pause
   1b390:	mov    esi,DWORD PTR [rdx]
   1b392:	test   sil,0x6
   1b396:	jne    1b38e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1fe>
   1b398:	nop    DWORD PTR [rax+rax*1+0x0]
   1b3a0:	mov    rax,QWORD PTR [rdx+0x18]
   1b3a4:	mov    ecx,0x4
   1b3a9:	mov    edi,eax
   1b3ab:	and    edi,0xf
   1b3ae:	je     1b3b7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x227>
   1b3b0:	lea    ecx,[rdi*4+0x0]
   1b3b7:	shr    rax,cl
   1b3ba:	mov    rcx,QWORD PTR [r13+0x8]
   1b3be:	and    eax,0xf
   1b3c1:	movzx  edi,BYTE PTR [rdx+rax*1+0x6]
   1b3c6:	mov    rax,QWORD PTR [rdx+rax*8+0x20]
   1b3cb:	cmp    rax,rcx
   1b3ce:	je     1b675 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x4e5>
   1b3d4:	cmp    rcx,rax
   1b3d7:	sbb    eax,eax
   1b3d9:	or     eax,0x1
   1b3dc:	mov    ecx,DWORD PTR [rdx]
   1b3de:	xor    esi,ecx
   1b3e0:	cmp    esi,0x1
   1b3e3:	ja     1b387 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1f7>
   1b3e5:	mov    ecx,DWORD PTR [rsp+0xc]
   1b3e9:	and    ecx,0x20000000
   1b3ef:	test   eax,eax
   1b3f1:	jle    1b323 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x193>
   1b3f7:	test   ecx,ecx
   1b3f9:	jne    1b43b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x2ab>
   1b3fb:	mov    rax,QWORD PTR [rdx+0x118]
   1b402:	mov    rcx,rdx
   1b405:	mov    rdx,rax
   1b408:	and    rdx,0xfffffffffffffffe
   1b40c:	cmp    rax,0x1
   1b410:	jbe    1b669 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x4d9>
   1b416:	mov    rax,QWORD PTR [rdx+0x20]
   1b41a:	cmp    QWORD PTR [r13+0x8],rax
   1b41e:	jae    1b42a <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x29a>
   1b420:	jmp    1b669 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x4d9>
   1b425:	nop    DWORD PTR [rax]
   1b428:	pause
   1b42a:	mov    eax,DWORD PTR [rdx]
   1b42c:	test   al,0x6
   1b42e:	jne    1b428 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x298>
   1b430:	mov    DWORD PTR [rsp+0xc],eax
   1b434:	test   eax,0x20000000
   1b439:	je     1b3fb <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x26b>
   1b43b:	mov    QWORD PTR [r13+0x0],rdx
   1b43f:	jmp    1b1c2 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x32>
   1b444:	nop    DWORD PTR [rax+0x0]
   1b448:	movzx  r11d,BYTE PTR [r14+rax*1+0x6]
   1b44e:	cmp    r10d,r11d
   1b451:	js     1b380 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x1f0>
   1b457:	jne    1b25b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xcb>
   1b45d:	mov    DWORD PTR [r13+0x2c],ecx
   1b461:	add    ecx,0x13
   1b464:	mov    DWORD PTR [r13+0x28],edx
   1b468:	movzx  edx,BYTE PTR [r14+rax*1+0x6]
   1b46e:	movsxd rcx,ecx
   1b471:	mov    rsi,QWORD PTR [r14+rcx*8]
   1b475:	test   dl,dl
   1b477:	js     1b550 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x3c0>
   1b47d:	prefetcht0 BYTE PTR [rsi]
   1b480:	cmp    edx,0x3f
   1b483:	jle    1b5c0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x430>
   1b489:	cmp    edx,0x40
   1b48c:	jne    1b56a <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x3da>
   1b492:	mov    rdx,QWORD PTR [r14+0x110]
   1b499:	test   rdx,rdx
   1b49c:	je     1b5d0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x440>
   1b4a2:	movzx  ecx,WORD PTR [rdx+rax*4+0x6]
   1b4a7:	movzx  eax,WORD PTR [rdx+rax*4+0x4]
   1b4ac:	lea    rdi,[rdx+rax*1]
   1b4b0:	sub    r10d,0x8
   1b4b4:	mov    rsi,QWORD PTR [r13+0x18]
   1b4b8:	xor    eax,eax
   1b4ba:	cmp    r10d,ecx
   1b4bd:	jne    1b270 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xe0>
   1b4c3:	cmp    ecx,0x8
   1b4c6:	jg     1b650 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x4c0>
   1b4cc:	mov    rax,QWORD PTR [rdi]
   1b4cf:	xor    rax,QWORD PTR [rsi+0x8]
   1b4d3:	test   ecx,ecx
   1b4d5:	je     1b5c0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x430>
   1b4db:	mov    edx,0x8
   1b4e0:	sub    edx,ecx
   1b4e2:	lea    ecx,[rdx*8+0x0]
   1b4e9:	shl    rax,cl
   1b4ec:	test   rax,rax
   1b4ef:	sete   al
   1b4f2:	movzx  eax,al
   1b4f5:	jmp    1b270 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xe0>
   1b4fa:	nop    WORD PTR [rax+rax*1+0x0]
   1b500:	pause
   1b502:	mov    edx,DWORD PTR [r14]
   1b505:	jmp    1b278 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xe8>
   1b50a:	nop    WORD PTR [rax+rax*1+0x0]
   1b510:	lea    eax,[rdx*4+0x0]
   1b517:	and    eax,0x8
   1b51a:	add    eax,edx
   1b51c:	and    eax,0xeffffff8
   1b521:	jmp    1b2eb <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x15b>
   1b526:	cs nop WORD PTR [rax+rax*1+0x0]
   1b530:	je     1b203 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x73>
   1b536:	mov    rax,QWORD PTR [rdx+0x110]
   1b53d:	prefetcht0 BYTE PTR [rax]
   1b540:	prefetcht0 BYTE PTR [rax+0x40]
   1b544:	jmp    1b203 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x73>
   1b549:	nop    DWORD PTR [rax+0x0]
   1b550:	prefetcht0 BYTE PTR [rsi]
   1b553:	prefetcht0 BYTE PTR [rsi+0x40]
   1b557:	prefetcht0 BYTE PTR [rsi+0x80]
   1b55e:	prefetcht0 BYTE PTR [rsi+0xc0]
   1b565:	cmp    dl,0x80
   1b568:	je     1b5a0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x410>
   1b56a:	mov    rax,QWORD PTR [rsp+0x18]
   1b56f:	sub    rax,QWORD PTR fs:0x28
   1b578:	jne    1b87f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x6ef>
   1b57e:	xor    ecx,ecx
   1b580:	lea    rdx,[rip+0x409ce]        # 5bf55 <_IO_stdin_used+0x1f55>
   1b587:	mov    esi,0x18a
   1b58c:	lea    rdi,[rip+0x409dc]        # 5bf6f <_IO_stdin_used+0x1f6f>
   1b593:	call   590f0 <_Z26fail_masstree_preconditionPKciS0_S0_>
   1b598:	nop    DWORD PTR [rax+rax*1+0x0]
   1b5a0:	mov    DWORD PTR [r13+0x38],0xfffffff8
   1b5a8:	mov    eax,DWORD PTR [rsp+0xc]
   1b5ac:	xor    eax,DWORD PTR [r14]
   1b5af:	cmp    eax,0x1
   1b5b2:	jbe    1b5ef <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x45f>
   1b5b4:	mov    r14,QWORD PTR [r13+0x0]
   1b5b8:	jmp    1b274 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xe4>
   1b5bd:	nop    DWORD PTR [rax]
   1b5c0:	mov    eax,0x1
   1b5c5:	jmp    1b270 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xe0>
   1b5ca:	nop    WORD PTR [rax+rax*1+0x0]
   1b5d0:	movzx  ecx,BYTE PTR [r14+rax*2+0x13b]
   1b5d9:	movzx  eax,BYTE PTR [r14+rax*2+0x13a]
   1b5e2:	lea    rdi,[r14+rax*1+0x138]
   1b5ea:	jmp    1b4b0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x320>
   1b5ef:	test   DWORD PTR [rsi],0x40000000
   1b5f5:	je     1b5b4 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x424>
   1b5f7:	mov    ecx,DWORD PTR [r13+0x38]
   1b5fb:	xor    eax,eax
   1b5fd:	mov    edx,ecx
   1b5ff:	add    ecx,DWORD PTR [r13+0x10]
   1b603:	neg    edx
   1b605:	mov    DWORD PTR [r13+0x10],ecx
   1b609:	movsxd rdx,edx
   1b60c:	add    rdx,QWORD PTR [r13+0x18]
   1b610:	mov    QWORD PTR [r13+0x18],rdx
   1b614:	test   ecx,ecx
   1b616:	jle    1b63b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x4ab>
   1b618:	cmp    ecx,0x7
   1b61b:	jg     1b7af <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x61f>
   1b621:	mov    eax,0x8
   1b626:	sub    eax,ecx
   1b628:	movsxd rcx,eax
   1b62b:	sub    rdx,rcx
   1b62e:	lea    ecx,[rax*8+0x0]
   1b635:	mov    rax,QWORD PTR [rdx]
   1b638:	shr    rax,cl
   1b63b:	bswap  rax
   1b63e:	mov    rbx,rsi
   1b641:	mov    QWORD PTR [r13+0x8],rax
   1b645:	jmp    1b1c2 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x32>
   1b64a:	nop    WORD PTR [rax+rax*1+0x0]
   1b650:	add    rsi,0x8
   1b654:	movsxd rdx,ecx
   1b657:	call   58a0 <memcmp@plt>
   1b65c:	test   eax,eax
   1b65e:	sete   al
   1b661:	movzx  eax,al
   1b664:	jmp    1b270 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0xe0>
   1b669:	mov    QWORD PTR [r13+0x0],rcx
   1b66d:	mov    rdx,rcx
   1b670:	jmp    1b1e0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x50>
   1b675:	mov    eax,DWORD PTR [r13+0x10]
   1b679:	cmp    eax,0x8
   1b67c:	jle    1b6c6 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x536>
   1b67e:	xor    eax,eax
   1b680:	cmp    edi,0x8
   1b683:	setle  al
   1b686:	jmp    1b3dc <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x24c>
   1b68b:	mov    eax,DWORD PTR [r13+0x38]
   1b68f:	test   eax,eax
   1b691:	js     1b6cd <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x53d>
   1b693:	cmp    BYTE PTR [r14+0x5],0x2
   1b698:	je     1b754 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x5c4>
   1b69e:	test   eax,eax
   1b6a0:	setne  al
   1b6a3:	mov    rdx,QWORD PTR [rsp+0x18]
   1b6a8:	sub    rdx,QWORD PTR fs:0x28
   1b6b1:	jne    1b87f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x6ef>
   1b6b7:	add    rsp,0x28
   1b6bb:	pop    rbx
   1b6bc:	pop    rbp
   1b6bd:	pop    r12
   1b6bf:	pop    r13
   1b6c1:	pop    r14
   1b6c3:	pop    r15
   1b6c5:	ret
   1b6c6:	sub    eax,edi
   1b6c8:	jmp    1b3dc <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x24c>
   1b6cd:	mov    ecx,eax
   1b6cf:	add    eax,DWORD PTR [r13+0x10]
   1b6d3:	xor    edx,edx
   1b6d5:	neg    ecx
   1b6d7:	mov    DWORD PTR [r13+0x10],eax
   1b6db:	movsxd rcx,ecx
   1b6de:	add    rcx,QWORD PTR [r13+0x18]
   1b6e2:	mov    QWORD PTR [r13+0x18],rcx
   1b6e6:	test   eax,eax
   1b6e8:	jle    1b711 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x581>
   1b6ea:	cmp    eax,0x7
   1b6ed:	jg     1b7a7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE11find_lockedER10threadinfo+0x617>
   1b6f3:	mov    edx,0x8
   1b6f8:	sub    edx,eax
   1b6fa:	mov    rax,rcx
   1b6fd:	movsxd rsi,edx
