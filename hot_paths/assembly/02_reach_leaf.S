mttest:     file format elf64-x86-64


Disassembly of section .text:

000000000000afb0 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0>:
    afb0:	push   rbp
    afb1:	mov    r9,rdi
    afb4:	mov    r10,rsi
    afb7:	mov    r11,rdx
    afba:	push   rbx
    afbb:	sub    rsp,0x38
    afbf:	mov    rax,QWORD PTR fs:0x28
    afc8:	mov    QWORD PTR [rsp+0x28],rax
    afcd:	xor    eax,eax
    afcf:	mov    QWORD PTR [rsp+0x10],r9
    afd4:	mov    eax,DWORD PTR [r9]
    afd7:	mov    rdx,r9
    afda:	jmp    afe4 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x34>
    afdc:	nop    DWORD PTR [rax+0x0]
    afe0:	pause
    afe2:	mov    eax,DWORD PTR [rdx]
    afe4:	test   al,0x6
    afe6:	jne    afe0 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x30>
    afe8:	mov    DWORD PTR [rsp+0x8],eax
    afec:	test   eax,0x40000000
    aff1:	jne    b029 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x79>
    aff3:	mov    rcx,QWORD PTR [rsp+0x10]
    aff8:	mov    eax,DWORD PTR [rcx]
    affa:	test   eax,eax
    affc:	js     b018 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x68>
    affe:	mov    rdx,QWORD PTR [rcx+0x108]
    b005:	test   rdx,rdx
    b008:	je     b024 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x74>
    b00a:	mov    eax,DWORD PTR [rdx]
    b00c:	mov    QWORD PTR [rsp+0x10],rdx
    b011:	jmp    afe4 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x34>
    b013:	nop    DWORD PTR [rax+rax*1+0x0]
    b018:	mov    rdx,QWORD PTR [rcx+0x128]
    b01f:	test   rdx,rdx
    b022:	jne    b00a <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x5a>
    b024:	mov    rdx,rcx
    b027:	jmp    b00c <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x5c>
    b029:	xor    ebx,ebx
    b02b:	xor    r8d,r8d
    b02e:	mov    rcx,QWORD PTR [rsp+r8*8+0x10]
    b033:	test   eax,eax
    b035:	js     b0e1 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x131>
    b03b:	nop    DWORD PTR [rax+rax*1+0x0]
    b040:	prefetcht0 BYTE PTR [rcx+0x40]
    b044:	prefetcht0 BYTE PTR [rcx+0x80]
    b04b:	prefetcht0 BYTE PTR [rcx+0xc0]
    b052:	movzx  esi,BYTE PTR [rcx+0x4]
    b056:	xor    edx,edx
    b058:	cmp    esi,edx
    b05a:	jle    b088 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0xd8>
    b05c:	lea    eax,[rsi+rdx*1]
    b05f:	mov    rbp,QWORD PTR [r10]
    b062:	sar    eax,1
    b064:	movsxd rdi,eax
    b067:	mov    rdi,QWORD PTR [rcx+rdi*8+0x10]
    b06c:	cmp    rdi,rbp
    b06f:	je     b110 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x160>
    b075:	cmp    rbp,rdi
    b078:	jb     b120 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x170>
    b07e:	lea    edx,[rax+0x1]
    b081:	cmp    esi,edx
    b083:	jg     b05c <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0xac>
    b085:	nop    DWORD PTR [rax]
    b088:	mov    edi,ebx
    b08a:	movsxd rdx,edx
    b08d:	mov    rdx,QWORD PTR [rcx+rdx*8+0x88]
    b095:	xor    edi,0x1
    b098:	mov    esi,edi
    b09a:	mov    QWORD PTR [rsp+rsi*8+0x10],rdx
    b09f:	test   rdx,rdx
    b0a2:	jne    b0b2 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x102>
    b0a4:	jmp    afcf <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1f>
    b0a9:	nop    DWORD PTR [rax+0x0]
    b0b0:	pause
    b0b2:	mov    eax,DWORD PTR [rdx]
    b0b4:	test   al,0x6
    b0b6:	jne    b0b0 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x100>
    b0b8:	mov    DWORD PTR [rsp+rsi*4+0x8],eax
    b0bc:	mov    eax,DWORD PTR [rsp+r8*4+0x8]
    b0c1:	mov    edx,DWORD PTR [rcx]
    b0c3:	xor    eax,edx
    b0c5:	cmp    eax,0x1
    b0c8:	ja     b130 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x180>
    b0ca:	mov    eax,DWORD PTR [rsp+rsi*4+0x8]
    b0ce:	mov    r8d,edi
    b0d1:	mov    rcx,QWORD PTR [rsp+r8*8+0x10]
    b0d6:	mov    rbx,r8
    b0d9:	test   eax,eax
    b0db:	jns    b040 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x90>
    b0e1:	mov    eax,DWORD PTR [rsp+r8*4+0x8]
    b0e6:	mov    DWORD PTR [r11],eax
    b0e9:	mov    rax,QWORD PTR [rsp+0x28]
    b0ee:	sub    rax,QWORD PTR fs:0x28
    b0f7:	jne    b1b4 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x204>
    b0fd:	add    rsp,0x38
    b101:	mov    rax,rcx
    b104:	pop    rbx
    b105:	pop    rbp
    b106:	ret
    b107:	nop    WORD PTR [rax+rax*1+0x0]
    b110:	lea    edx,[rax+0x1]
    b113:	jmp    b088 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0xd8>
    b118:	nop    DWORD PTR [rax+rax*1+0x0]
    b120:	mov    esi,eax
    b122:	jmp    b058 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0xa8>
    b127:	nop    WORD PTR [rax+rax*1+0x0]
    b130:	mov    eax,DWORD PTR [rsp+r8*4+0x8]
    b135:	test   dl,0x6
    b138:	je     b149 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x199>
    b13a:	nop    WORD PTR [rax+rax*1+0x0]
    b140:	pause
    b142:	mov    edx,DWORD PTR [rcx]
    b144:	test   dl,0x6
    b147:	jne    b140 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x190>
    b149:	mov    DWORD PTR [rsp+r8*4+0x8],edx
    b14e:	xor    eax,edx
    b150:	cmp    eax,0x1ff
    b155:	ja     b161 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1b1>
    b157:	mov    eax,DWORD PTR [rsp+r8*4+0x8]
    b15c:	jmp    b02e <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x7e>
    b161:	mov    esi,DWORD PTR [rsp+r8*4+0x8]
    b166:	movzx  eax,BYTE PTR [rcx+0x4]
    b16a:	mov    edx,0x1
    b16f:	test   eax,eax
    b171:	je     b18f <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1df>
    b173:	sub    eax,0x1
    b176:	mov    rdi,QWORD PTR [r10]
    b179:	xor    edx,edx
    b17b:	cdqe
    b17d:	mov    rax,QWORD PTR [rcx+rax*8+0x10]
    b182:	cmp    rax,rdi
    b185:	je     b18f <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1df>
    b187:	cmp    rdi,rax
    b18a:	sbb    edx,edx
    b18c:	or     edx,0x1
    b18f:	mov    eax,DWORD PTR [rcx]
    b191:	xor    esi,eax
    b193:	cmp    esi,0x1
    b196:	ja     b1ac <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1fc>
    b198:	cmp    edx,0x1
    b19b:	jne    b157 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1a7>
    b19d:	jmp    afcf <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1f>
    b1a2:	nop    WORD PTR [rax+rax*1+0x0]
    b1a8:	pause
    b1aa:	mov    eax,DWORD PTR [rcx]
    b1ac:	test   al,0x6
    b1ae:	jne    b1a8 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1f8>
    b1b0:	mov    esi,eax
    b1b2:	jmp    b166 <_ZNK8Masstree9node_baseINS_26default_query_table_paramsEE10reach_leafERKNS_3keyImEER11nodeversionI22nodeversion_parametersIjEER10threadinfo.isra.0+0x1b6>
    b1b4:	call   5ac0 <__stack_chk_fail@plt>
    b1b9:	nop    DWORD PTR [rax+0x0]

000000000000b1c0 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0>:
    b1c0:	cmp    rdi,rsi
    b1c3:	je     b266 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0xa6>
    b1c9:	push   r14
    b1cb:	mov    r14,rsi
    b1ce:	push   r13
    b1d0:	mov    r13,rdi
    b1d3:	push   r12
    b1d5:	push   rbp
    b1d6:	push   rbx
    b1d7:	lea    rbx,[rdi+0x4]
    b1db:	cmp    rsi,rbx
    b1de:	je     b254 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x94>
    b1e0:	mov    r12d,0x4
    b1e6:	jmp    b21a <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x5a>
    b1e8:	nop    DWORD PTR [rax+rax*1+0x0]
    b1f0:	mov    rdx,rbx
    b1f3:	sub    rdx,r13
    b1f6:	cmp    rdx,0x4
    b1fa:	jle    b260 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0xa0>
    b1fc:	mov    rdi,r12
    b1ff:	mov    rsi,r13
    b202:	sub    rdi,rdx
    b205:	add    rdi,rbx
    b208:	call   5d40 <memmove@plt>
    b20d:	add    rbx,0x4
    b211:	mov    DWORD PTR [r13+0x0],ebp
    b215:	cmp    r14,rbx
    b218:	je     b254 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x94>
    b21a:	mov    ebp,DWORD PTR [rbx]
    b21c:	mov    eax,DWORD PTR [r13+0x0]
    b220:	mov    rcx,rbx
    b223:	cmp    ebp,eax
    b225:	jl     b1f0 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x30>
    b227:	mov    edx,DWORD PTR [rbx-0x4]
    b22a:	lea    rax,[rbx-0x4]
    b22e:	cmp    ebp,edx
    b230:	jge    b249 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x89>
    b232:	nop    WORD PTR [rax+rax*1+0x0]
    b238:	mov    DWORD PTR [rax+0x4],edx
    b23b:	mov    rcx,rax
    b23e:	mov    edx,DWORD PTR [rax-0x4]
    b241:	sub    rax,0x4
    b245:	cmp    ebp,edx
    b247:	jl     b238 <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x78>
    b249:	add    rbx,0x4
    b24d:	mov    DWORD PTR [rcx],ebp
    b24f:	cmp    r14,rbx
    b252:	jne    b21a <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x5a>
    b254:	pop    rbx
    b255:	pop    rbp
    b256:	pop    r12
    b258:	pop    r13
    b25a:	pop    r14
    b25c:	ret
    b25d:	nop    DWORD PTR [rax]
    b260:	jne    b20d <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x4d>
    b262:	mov    DWORD PTR [rbx],eax
    b264:	jmp    b20d <_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEENS0_5__ops15_Iter_less_iterEEvT_S9_T0_.isra.0+0x4d>
    b266:	ret
    b267:	nop    WORD PTR [rax+rax*1+0x0]

000000000000b270 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0>:
    b270:	mov    rax,rsi
    b273:	mov    rcx,rsi
    b276:	sub    rax,rdi
    b279:	cmp    rax,0x80
    b27f:	jle    b44b <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x1db>
    b285:	push   r13
    b287:	mov    r13,rdx
    b28a:	push   r12
    b28c:	mov    r12,rdi
    b28f:	push   rbp
    b290:	lea    rbp,[rdi+0x8]
    b294:	push   rbx
    b295:	sub    rsp,0x8
    b299:	test   rdx,rdx
    b29c:	je     b3db <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x16b>
    b2a2:	movupd xmm0,XMMWORD PTR [r12]
    b2a8:	sar    rax,0x4
    b2ac:	sub    r13,0x1
    b2b0:	movsd  xmm3,QWORD PTR [rsi-0x8]
    b2b5:	lea    rax,[r12+rax*8]
    b2b9:	movsd  xmm2,QWORD PTR [rax]
    b2bd:	movapd xmm5,xmm0
    b2c1:	movapd xmm4,xmm0
    b2c5:	unpckhpd xmm5,xmm5
    b2c9:	shufpd xmm4,xmm0,0x1
    b2ce:	comisd xmm2,xmm5
    b2d2:	movapd xmm1,xmm5
    b2d6:	jbe    b362 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xf2>
    b2dc:	comisd xmm3,xmm2
    b2e0:	ja     b3ba <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x14a>
    b2e6:	comisd xmm3,xmm5
    b2ea:	ja     b36e <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xfe>
    b2f0:	movups XMMWORD PTR [r12],xmm4
    b2f5:	movsd  xmm2,QWORD PTR [rsi-0x8]
    b2fa:	mov    rdx,rsi
    b2fd:	mov    rcx,rbp
    b300:	comisd xmm1,xmm0
    b304:	lea    rax,[rcx+0x8]
    b308:	jbe    b321 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xb1>
    b30a:	nop    WORD PTR [rax+rax*1+0x0]
    b310:	mov    rcx,rax
    b313:	movsd  xmm0,QWORD PTR [rax]
    b317:	add    rax,0x8
    b31b:	comisd xmm1,xmm0
    b31f:	ja     b310 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xa0>
    b321:	sub    rdx,0x8
    b325:	comisd xmm2,xmm1
    b329:	mov    rbx,rcx
    b32c:	jbe    b33f <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xcf>
    b32e:	xchg   ax,ax
    b330:	movsd  xmm2,QWORD PTR [rdx-0x8]
    b335:	sub    rdx,0x8
    b339:	comisd xmm2,xmm1
    b33d:	ja     b330 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xc0>
    b33f:	cmp    rbx,rdx
    b342:	jae    b390 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x120>
    b344:	movsd  QWORD PTR [rbx],xmm2
    b348:	lea    rcx,[rbx+0x8]
    b34c:	movsd  xmm2,QWORD PTR [rdx-0x8]
    b351:	movsd  QWORD PTR [rdx],xmm0
    b355:	movsd  xmm1,QWORD PTR [r12]
    b35b:	movsd  xmm0,QWORD PTR [rbx+0x8]
    b360:	jmp    b300 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x90>
    b362:	comisd xmm3,xmm5
    b366:	ja     b2f0 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x80>
    b368:	comisd xmm3,xmm2
    b36c:	jbe    b3ba <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x14a>
    b36e:	movsd  QWORD PTR [r12],xmm3
    b374:	movapd xmm2,xmm0
    b378:	movsd  QWORD PTR [rsi-0x8],xmm0
    b37d:	movsd  xmm1,QWORD PTR [r12]
    b383:	movsd  xmm0,QWORD PTR [r12+0x8]
    b38a:	jmp    b2fa <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x8a>
    b38f:	nop
    b390:	mov    rdx,r13
    b393:	mov    rdi,rbx
    b396:	call   b270 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0>
    b39b:	mov    rax,rbx
    b39e:	sub    rax,r12
    b3a1:	cmp    rax,0x80
    b3a7:	jle    b440 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x1d0>
    b3ad:	test   r13,r13
    b3b0:	je     b3de <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x16e>
    b3b2:	mov    rsi,rbx
    b3b5:	jmp    b2a2 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x32>
    b3ba:	movsd  QWORD PTR [r12],xmm2
    b3c0:	movsd  QWORD PTR [rax],xmm0
    b3c4:	movsd  xmm0,QWORD PTR [r12+0x8]
    b3cb:	movsd  xmm1,QWORD PTR [r12]
    b3d1:	movsd  xmm2,QWORD PTR [rsi-0x8]
    b3d6:	jmp    b2fa <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x8a>
    b3db:	mov    rbx,rcx
    b3de:	sar    rax,0x3
    b3e2:	lea    r13,[rax-0x2]
    b3e6:	mov    rbp,rax
    b3e9:	sar    r13,1
    b3ec:	jmp    b3f2 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x182>
    b3ee:	sub    r13,0x1
    b3f2:	movsd  xmm0,QWORD PTR [r12+r13*8]
    b3f8:	mov    rdx,rbp
    b3fb:	mov    rsi,r13
    b3fe:	mov    rdi,r12
    b401:	call   acf0 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEldNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0>
    b406:	test   r13,r13
    b409:	jne    b3ee <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x17e>
    b40b:	sub    rbx,0x8
    b40f:	nop
    b410:	mov    rbp,rbx
    b413:	movsd  xmm1,QWORD PTR [r12]
    b419:	movsd  xmm0,QWORD PTR [rbx]
    b41d:	xor    esi,esi
    b41f:	sub    rbp,r12
    b422:	mov    rdi,r12
    b425:	sub    rbx,0x8
    b429:	mov    rdx,rbp
    b42c:	movsd  QWORD PTR [rbx+0x8],xmm1
    b431:	sar    rdx,0x3
    b435:	call   acf0 <_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEldNS0_5__ops15_Iter_less_iterEEvT_T0_SA_T1_T2_.isra.0>
    b43a:	cmp    rbp,0x8
    b43e:	jg     b410 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x1a0>
    b440:	add    rsp,0x8
    b444:	pop    rbx
    b445:	pop    rbp
    b446:	pop    r12
    b448:	pop    r13
    b44a:	ret
    b44b:	ret
    b44c:	nop    DWORD PTR [rax+0x0]

000000000000b450 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0>:
    b450:	mov    rax,rsi
    b453:	sub    rax,rdi
    b456:	cmp    rax,0x40
    b45a:	jle    b5ee <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x19e>
    b460:	push   r13
    b462:	mov    r9,rsi
    b465:	mov    r13,rdx
    b468:	push   r12
    b46a:	mov    r12,rdi
    b46d:	push   rbp
    b46e:	push   rbx
    b46f:	lea    rbx,[rdi+0x4]
    b473:	sub    rsp,0x8
    b477:	test   rdx,rdx
    b47a:	je     b583 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x133>
    b480:	sar    rax,0x3
    b484:	movq   xmm0,QWORD PTR [r12]
    b48a:	mov    edx,DWORD PTR [r9-0x4]
    b48e:	sub    r13,0x1
    b492:	lea    rdi,[r12+rax*4]
    b496:	mov    eax,DWORD PTR [rdi]
    b498:	pshufd xmm2,xmm0,0xe5
    b49d:	movd   esi,xmm2
    b4a1:	pshufd xmm1,xmm0,0xe1
    b4a6:	movd   ecx,xmm0
    b4aa:	cmp    eax,esi
    b4ac:	jle    b51c <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xcc>
    b4ae:	cmp    eax,edx
    b4b0:	jl     b56b <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x11b>
    b4b6:	cmp    esi,edx
    b4b8:	jl     b524 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xd4>
    b4ba:	movq   QWORD PTR [r12],xmm1
    b4c0:	mov    edi,DWORD PTR [r9-0x4]
    b4c4:	mov    rdx,r9
    b4c7:	mov    rax,rbx
    b4ca:	cmp    esi,ecx
    b4cc:	jle    b517 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xc7>
    b4ce:	xchg   ax,ax
    b4d0:	add    rax,0x4
    b4d4:	nop    DWORD PTR [rax+0x0]
    b4d8:	mov    r8,rax
    b4db:	mov    ecx,DWORD PTR [rax]
    b4dd:	add    rax,0x4
    b4e1:	cmp    ecx,esi
    b4e3:	jl     b4d8 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0x88>
    b4e5:	mov    rbp,r8
    b4e8:	sub    rdx,0x4
    b4ec:	cmp    esi,edi
    b4ee:	jge    b4fb <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xab>
    b4f0:	mov    edi,DWORD PTR [rdx-0x4]
    b4f3:	sub    rdx,0x4
    b4f7:	cmp    edi,esi
    b4f9:	jg     b4f0 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xa0>
    b4fb:	cmp    rbp,rdx
    b4fe:	jae    b540 <_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_.isra.0+0xf0>
