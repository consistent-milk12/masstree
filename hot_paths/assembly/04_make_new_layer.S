mttest:     file format elf64-x86-64


Disassembly of section .text:

000000000001d710 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo>:
   1d710:	endbr64
   1d714:	push   r15
   1d716:	mov    r15,rdi
   1d719:	push   r14
   1d71b:	push   r13
   1d71d:	mov    r13,rsi
   1d720:	push   r12
   1d722:	push   rbp
   1d723:	push   rbx
   1d724:	sub    rsp,0x78
   1d728:	mov    rdi,QWORD PTR [rdi]
   1d72b:	mov    rax,QWORD PTR fs:0x28
   1d734:	mov    QWORD PTR [rsp+0x68],rax
   1d739:	xor    eax,eax
   1d73b:	movsxd rax,DWORD PTR [r15+0x2c]
   1d73f:	movzx  edx,BYTE PTR [rdi+rax*1+0x6]
   1d744:	mov    rsi,rax
   1d747:	call   1bb50 <_ZNK8Masstree4leafINS_26default_query_table_paramsEE4ksufEii>
   1d74c:	mov    rbx,rdx
   1d74f:	movq   xmm0,rax
   1d754:	mov    r14d,edx
   1d757:	xor    edx,edx
   1d759:	punpcklqdq xmm0,xmm0
   1d75d:	test   ebx,ebx
   1d75f:	jle    1d76d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x5d>
   1d761:	cmp    ebx,0x7
   1d764:	jle    1da7e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x36e>
   1d76a:	mov    rdx,QWORD PTR [rax]
   1d76d:	lea    rax,[r15+0x8]
   1d771:	mov    rdi,QWORD PTR [r15+0x18]
   1d775:	mov    DWORD PTR [rsp+0x48],ebx
   1d779:	mov    QWORD PTR [rsp+0x10],rax
   1d77e:	mov    eax,DWORD PTR [r15+0x10]
   1d782:	lea    rcx,[rdi+0x8]
   1d786:	movaps XMMWORD PTR [rsp+0x50],xmm0
   1d78b:	lea    esi,[rax-0x8]
   1d78e:	mov    QWORD PTR [r15+0x18],rcx
   1d792:	xor    eax,eax
   1d794:	bswap  rdx
   1d797:	mov    DWORD PTR [r15+0x10],esi
   1d79b:	mov    QWORD PTR [rsp+0x40],rdx
   1d7a0:	test   esi,esi
   1d7a2:	jle    1d7ca <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0xba>
   1d7a4:	cmp    esi,0x7
   1d7a7:	jg     1daaa <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x39a>
   1d7ad:	mov    edi,0x8
   1d7b2:	mov    rax,rcx
   1d7b5:	sub    edi,esi
   1d7b7:	movsxd r9,edi
   1d7ba:	lea    ecx,[rdi*8+0x0]
   1d7c1:	sub    rax,r9
   1d7c4:	mov    rax,QWORD PTR [rax]
   1d7c7:	shr    rax,cl
   1d7ca:	bswap  rax
   1d7cd:	mov    rbp,QWORD PTR [r15]
   1d7d0:	mov    QWORD PTR [r15+0x8],rax
   1d7d4:	cmp    rdx,rax
   1d7d7:	je     1dae2 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x3d2>
   1d7dd:	sbb    r9d,r9d
   1d7e0:	mov    rbx,rbp
   1d7e3:	or     r9d,0x1
   1d7e7:	cmp    esi,0x8
   1d7ea:	jle    1dd59 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x649>
   1d7f0:	sub    esi,0x8
   1d7f3:	sub    r14d,0x8
   1d7f7:	mov    eax,0x0
   1d7fc:	cmovs  r14d,eax
   1d800:	xor    ecx,ecx
   1d802:	add    r14d,esi
   1d805:	lea    eax,[r14*8+0x0]
   1d80d:	sub    eax,r14d
   1d810:	lea    edx,[rax+0x20]
   1d813:	test   rbx,rbx
   1d816:	je     1d81f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x10f>
   1d818:	mov    rcx,QWORD PTR [rbx+0x130]
   1d81f:	cmp    edx,0x80
   1d825:	jle    1dab3 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x3a3>
   1d82b:	xor    edx,edx
   1d82d:	mov    r14d,0x6
   1d833:	mov    esi,0x7
   1d838:	mov    eax,0x1c0
   1d83d:	movsxd r14,r14d
   1d840:	lea    r10,[r14+0x8]
   1d844:	mov    r12,QWORD PTR [r13+r10*8+0x0]
   1d849:	test   r12,r12
   1d84c:	je     1dcf1 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x5e1>
   1d852:	mov    rsi,QWORD PTR [r12]
   1d856:	mov    QWORD PTR [r13+r14*8+0x40],rsi
   1d85b:	movabs rdi,0x123456789abcde0
   1d865:	mov    DWORD PTR [r12],0x80000000
   1d86d:	mov    BYTE PTR [r12+0x5],0x0
   1d873:	mov    QWORD PTR [r12+0x18],rdi
   1d878:	mov    QWORD PTR [r12+0x110],0x0
   1d884:	mov    QWORD PTR [r12+0x128],0x0
   1d890:	test   dl,dl
   1d892:	jne    1df11 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x801>
   1d898:	mov    edx,eax
   1d89a:	sar    edx,0x6
   1d89d:	sub    edx,0x5
   1d8a0:	mov    BYTE PTR [r12+0x4],dl
   1d8a5:	test   dl,dl
   1d8a7:	je     1d8ed <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x1dd>
   1d8a9:	mov    BYTE PTR [r12+0x138],0x20
   1d8b2:	sub    eax,0x39
   1d8b5:	pxor   xmm0,xmm0
   1d8b9:	mov    BYTE PTR [r12+0x139],al
   1d8c1:	xor    eax,eax
   1d8c3:	mov    QWORD PTR [r12+0x14a],0x0
   1d8cf:	mov    DWORD PTR [r12+0x152],0x0
   1d8db:	mov    WORD PTR [r12+0x156],ax
   1d8e4:	movups XMMWORD PTR [r12+0x13a],xmm0
   1d8ed:	pxor   xmm0,xmm0
   1d8f1:	mov    QWORD PTR [r12+0x130],rcx
   1d8f9:	mov    QWORD PTR [r12+0x20],0x0
   1d902:	mov    DWORD PTR [r12],0xc0000000
   1d90a:	movups XMMWORD PTR [r12+0x118],xmm0
   1d913:	lea    rax,[rsp+0x40]
   1d918:	test   r9d,r9d
   1d91b:	cmovns rax,QWORD PTR [rsp+0x10]
   1d921:	mov    QWORD PTR [r12+0x98],0x0
   1d92d:	mov    rdx,QWORD PTR [rax]
   1d930:	mov    QWORD PTR [r12+0x20],rdx
   1d935:	mov    edx,DWORD PTR [rax+0x8]
   1d938:	cmp    edx,0x8
   1d93b:	jg     1dc9c <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x58c>
   1d941:	mov    BYTE PTR [r12+0x6],dl
   1d946:	lea    rax,[rsp+0x40]
   1d94b:	test   r9d,r9d
   1d94e:	cmovs  rax,QWORD PTR [rsp+0x10]
   1d954:	mov    QWORD PTR [r12+0xa0],0x0
   1d960:	mov    rdi,rax
   1d963:	mov    QWORD PTR [rsp+0x10],rax
   1d968:	mov    rax,QWORD PTR [rax]
   1d96b:	mov    QWORD PTR [r12+0x28],rax
   1d970:	mov    eax,DWORD PTR [rdi+0x8]
   1d973:	cmp    eax,0x8
   1d976:	jg     1dc63 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x553>
   1d97c:	mov    BYTE PTR [r12+0x7],al
   1d981:	mov    rdx,QWORD PTR [r15]
   1d984:	movsxd rcx,DWORD PTR [r15+0x2c]
   1d988:	test   r9d,r9d
   1d98b:	mov    rdi,r12
   1d98e:	setg   al
   1d991:	mov    esi,DWORD PTR [r12]
   1d995:	mov    DWORD PTR [rsp],r9d
   1d999:	mov    rdx,QWORD PTR [rdx+rcx*8+0x98]
   1d9a1:	movzx  eax,al
   1d9a4:	mov    QWORD PTR [r12+rax*8+0x98],rdx
   1d9ac:	call   af40 <_ZN11nodeversionI22nodeversion_parametersIjEE4lockIN10threadinfo31accounting_relax_fence_functionEEES2_S2_T_.isra.0>
   1d9b1:	mov    r9d,DWORD PTR [rsp]
   1d9b5:	movabs rdx,0x123456789abcde01
   1d9bf:	movabs rax,0x23456789abcde11
   1d9c9:	test   r9d,r9d
   1d9cc:	cmovs  rax,rdx
   1d9d0:	mov    rdx,QWORD PTR [r15]
   1d9d3:	mov    QWORD PTR [r12+0x18],rax
   1d9d8:	mov    eax,DWORD PTR [rdx]
   1d9da:	test   al,0x1
   1d9dc:	je     1df43 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x833>
   1d9e2:	or     eax,0x2
   1d9e5:	mov    DWORD PTR [rdx],eax
   1d9e7:	cmp    QWORD PTR [r15],rbx
   1d9ea:	je     1d9f6 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x2e6>
   1d9ec:	mov    QWORD PTR [rbx+0x98],r12
   1d9f3:	mov    rbx,QWORD PTR [r15]
   1d9f6:	movsxd rax,DWORD PTR [r15+0x2c]
   1d9fa:	lea    rdx,[rax+0x13]
   1d9fe:	cmp    rbp,rbx
   1da01:	je     1dce8 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x5d8>
   1da07:	mov    QWORD PTR [rbx+rdx*8],rbp
   1da0b:	mov    BYTE PTR [rbx+rax*1+0x6],0x80
   1da10:	mov    rbx,QWORD PTR [r15]
   1da13:	mov    esi,DWORD PTR [rbx]
   1da15:	mov    rdi,rbx
   1da18:	mov    DWORD PTR [rsp+0x3c],esi
   1da1c:	test   sil,0x1
   1da20:	jne    1dcca <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x5ba>
   1da26:	mov    rax,QWORD PTR [rbx+0x18]
   1da2a:	shl    esi,0x4
   1da2d:	mov    DWORD PTR [rsp],r9d
   1da31:	and    eax,0xf
   1da34:	add    eax,esi
   1da36:	mov    DWORD PTR [r15+0x4c],eax
   1da3a:	mov    esi,DWORD PTR [rdi]
   1da3c:	call   174e0 <_ZN11nodeversionI22nodeversion_parametersIjEE6unlockES2_>
   1da41:	mov    edx,DWORD PTR [rsp]
   1da44:	mov    QWORD PTR [r15],r12
   1da47:	shr    edx,0x1f
   1da4a:	movd   xmm2,edx
   1da4e:	pshufd xmm0,xmm2,0xe0
   1da53:	movq   QWORD PTR [r15+0x28],xmm0
   1da59:	mov    rax,QWORD PTR [rsp+0x68]
   1da5e:	sub    rax,QWORD PTR fs:0x28
   1da67:	jne    1df6d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x85d>
   1da6d:	add    rsp,0x78
   1da71:	xor    eax,eax
   1da73:	pop    rbx
   1da74:	pop    rbp
   1da75:	pop    r12
   1da77:	pop    r13
   1da79:	pop    r14
   1da7b:	pop    r15
   1da7d:	ret
   1da7e:	movsxd rdx,ebx
   1da81:	lea    rdi,[rsp+0x60]
   1da86:	mov    rsi,rax
   1da89:	movaps XMMWORD PTR [rsp],xmm0
   1da8d:	mov    QWORD PTR [rsp+0x60],0x0
   1da96:	call   59e0 <memcpy@plt>
   1da9b:	mov    rdx,QWORD PTR [rsp+0x60]
   1daa0:	movdqa xmm0,XMMWORD PTR [rsp]
   1daa5:	jmp    1d76d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x5d>
   1daaa:	mov    rax,QWORD PTR [rdi+0x8]
   1daae:	jmp    1d7ca <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0xba>
   1dab3:	lea    edx,[rax+0x158]
   1dab9:	movsxd rax,edx
   1dabc:	and    edx,0x3f
   1dabf:	jne    1dedf <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x7cf>
   1dac5:	lea    r14,[rax+0x3f]
   1dac9:	shr    r14,0x6
   1dacd:	cmp    rax,0x1fff
   1dad3:	mov    esi,r14d
   1dad6:	seta   dl
   1dad9:	sub    r14d,0x1
   1dadd:	jmp    1d83d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x12d>
   1dae2:	cmp    ebx,0x8
   1dae5:	jg     1dd86 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x676>
   1daeb:	mov    r9d,ebx
   1daee:	sub    r9d,esi
   1daf1:	jne    1df3b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x82b>
   1daf7:	movabs r12,0x123456789abcde01
   1db01:	mov    r14,rbp
   1db04:	nop    DWORD PTR [rax+0x0]
   1db08:	xor    eax,eax
   1db0a:	test   r14,r14
   1db0d:	je     1db16 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x406>
   1db0f:	mov    rax,QWORD PTR [r14+0x130]
   1db16:	mov    rbx,QWORD PTR [r13+0x60]
   1db1a:	test   rbx,rbx
   1db1d:	je     1dde0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x6d0>
   1db23:	mov    rdx,QWORD PTR [rbx]
   1db26:	mov    QWORD PTR [r13+0x60],rdx
   1db2a:	movdqa xmm1,XMMWORD PTR [rip+0x3ffce]        # 5db00 <_ZTSN8Masstree21gc_layer_rcu_callbackINS_26default_query_table_paramsEEE+0x140>
   1db32:	pxor   xmm0,xmm0
   1db36:	xor    edx,edx
   1db38:	mov    QWORD PTR [rbx+0x130],rax
   1db3f:	mov    DWORD PTR [rbx],0xc0000000
   1db45:	mov    WORD PTR [rbx+0x4],dx
   1db49:	movups XMMWORD PTR [rbx+0x110],xmm0
   1db50:	movups XMMWORD PTR [rbx+0x120],xmm0
   1db57:	movups XMMWORD PTR [rbx+0x18],xmm1
   1db5b:	mov    edx,DWORD PTR [rsp+0x48]
   1db5f:	cmp    edx,0x8
   1db62:	jle    1dec0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x7b0>
   1db68:	mov    rax,QWORD PTR [rsp+0x40]
   1db6d:	mov    BYTE PTR [rbx+0x6],0x80
   1db71:	mov    QWORD PTR [rbx+0x20],rax
   1db75:	cmp    QWORD PTR [r15],rbp
   1db78:	je     1de10 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x700>
   1db7e:	mov    QWORD PTR [r14+0x98],rbx
   1db85:	mov    esi,DWORD PTR [rbx]
   1db87:	mov    QWORD PTR [rbx+0x18],r12
   1db8b:	mov    eax,0x1
   1db90:	mov    DWORD PTR [rsp+0x3c],esi
   1db94:	test   sil,0x1
   1db98:	jne    1ddc0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x6b0>
   1db9e:	shl    esi,0x4
   1dba1:	lea    r14d,[rsi+rax*1]
   1dba5:	mov    rax,QWORD PTR [r15+0x58]
   1dba9:	cmp    rax,QWORD PTR [r15+0x60]
   1dbad:	je     1de60 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x750>
   1dbb3:	mov    rcx,QWORD PTR [rsp+0x50]
   1dbb8:	mov    DWORD PTR [rax+0x8],r14d
   1dbbc:	add    rax,0x10
   1dbc0:	lea    r14d,[rdx-0x8]
   1dbc4:	mov    QWORD PTR [rax-0x10],rbx
   1dbc8:	mov    QWORD PTR [r15+0x58],rax
   1dbcc:	lea    rax,[rcx+0x8]
   1dbd0:	mov    QWORD PTR [rsp+0x50],rax
   1dbd5:	mov    DWORD PTR [rsp+0x48],r14d
   1dbda:	cmp    r14d,0x7
   1dbde:	jg     1de20 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x710>
   1dbe4:	mov    ecx,0x8
   1dbe9:	sub    ecx,r14d
   1dbec:	movsxd rsi,ecx
   1dbef:	shl    ecx,0x3
   1dbf2:	sub    rax,rsi
   1dbf5:	mov    rax,QWORD PTR [rax]
   1dbf8:	shr    rax,cl
   1dbfb:	mov    rcx,QWORD PTR [r15+0x18]
   1dbff:	mov    edi,DWORD PTR [r15+0x10]
   1dc03:	bswap  rax
   1dc06:	lea    r9,[rcx+0x8]
   1dc0a:	lea    esi,[rdi-0x8]
   1dc0d:	mov    QWORD PTR [rsp+0x40],rax
   1dc12:	mov    QWORD PTR [r15+0x18],r9
   1dc16:	mov    DWORD PTR [r15+0x10],esi
   1dc1a:	test   esi,esi
   1dc1c:	jle    1dd40 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x630>
   1dc22:	cmp    esi,0x7
   1dc25:	jg     1de30 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x720>
   1dc2b:	mov    ecx,0x8
   1dc30:	sub    ecx,esi
   1dc32:	movsxd r10,ecx
   1dc35:	shl    ecx,0x3
   1dc38:	sub    r9,r10
   1dc3b:	mov    r9,QWORD PTR [r9]
   1dc3e:	shr    r9,cl
   1dc41:	mov    rcx,r9
   1dc44:	bswap  rcx
   1dc47:	mov    QWORD PTR [r15+0x8],rcx
   1dc4b:	cmp    rax,rcx
   1dc4e:	je     1de98 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x788>
   1dc54:	cmp    rax,rcx
   1dc57:	sbb    r9d,r9d
   1dc5a:	or     r9d,0x1
   1dc5e:	jmp    1d7e7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0xd7>
   1dc63:	mov    rax,rdi
   1dc66:	mov    r8,r13
   1dc69:	mov    esi,0x1
   1dc6e:	mov    DWORD PTR [rsp+0x18],r9d
   1dc73:	mov    BYTE PTR [r12+0x7],0x40
   1dc79:	mov    edi,DWORD PTR [rdi+0x8]
   1dc7c:	mov    rdx,QWORD PTR [rax+0x10]
   1dc80:	mov    DWORD PTR [rsp],edi
   1dc83:	lea    ecx,[rdi-0x8]
   1dc86:	mov    rdi,r12
   1dc89:	add    rdx,0x8
   1dc8d:	call   12170 <_ZN8Masstree4leafINS_26default_query_table_paramsEE11assign_ksufEiN4lcdf3StrEbR10threadinfo.constprop.0.isra.0>
   1dc92:	mov    r9d,DWORD PTR [rsp+0x18]
   1dc97:	jmp    1d981 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x271>
   1dc9c:	mov    BYTE PTR [r12+0x6],0x40
   1dca2:	mov    edi,DWORD PTR [rax+0x8]
   1dca5:	mov    r8,r13
   1dca8:	xor    esi,esi
   1dcaa:	mov    rdx,QWORD PTR [rax+0x10]
   1dcae:	mov    DWORD PTR [rsp],r9d
   1dcb2:	lea    ecx,[rdi-0x8]
   1dcb5:	mov    rdi,r12
   1dcb8:	add    rdx,0x8
   1dcbc:	call   12170 <_ZN8Masstree4leafINS_26default_query_table_paramsEE11assign_ksufEiN4lcdf3StrEbR10threadinfo.constprop.0.isra.0>
   1dcc1:	mov    r9d,DWORD PTR [rsp]
   1dcc5:	jmp    1d946 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x236>
   1dcca:	lea    rdi,[rsp+0x3c]
   1dccf:	mov    DWORD PTR [rsp],r9d
   1dcd3:	call   174e0 <_ZN11nodeversionI22nodeversion_parametersIjEE6unlockES2_>
   1dcd8:	mov    rdi,QWORD PTR [r15]
   1dcdb:	mov    esi,DWORD PTR [rsp+0x3c]
   1dcdf:	mov    r9d,DWORD PTR [rsp]
   1dce3:	jmp    1da26 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x316>
   1dce8:	mov    QWORD PTR [rbx+rdx*8],r12
   1dcec:	jmp    1da0b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x2fb>
   1dcf1:	mov    rdi,r13
   1dcf4:	mov    QWORD PTR [rsp+0x28],r10
   1dcf9:	mov    BYTE PTR [rsp+0x27],dl
   1dcfd:	mov    DWORD PTR [rsp+0x20],r9d
   1dd02:	mov    QWORD PTR [rsp+0x18],rax
   1dd07:	mov    QWORD PTR [rsp],rcx
   1dd0b:	call   59690 <_ZN10threadinfo11refill_poolEi>
   1dd10:	mov    r10,QWORD PTR [rsp+0x28]
   1dd15:	mov    rcx,QWORD PTR [rsp]
   1dd19:	mov    rax,QWORD PTR [rsp+0x18]
   1dd1e:	mov    r9d,DWORD PTR [rsp+0x20]
   1dd23:	mov    r12,QWORD PTR [r13+r10*8+0x0]
   1dd28:	movzx  edx,BYTE PTR [rsp+0x27]
   1dd2d:	test   r12,r12
   1dd30:	je     1d85b <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x14b>
   1dd36:	jmp    1d852 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x142>
   1dd3b:	nop    DWORD PTR [rax+rax*1+0x0]
   1dd40:	xor    ecx,ecx
   1dd42:	bswap  rcx
   1dd45:	mov    QWORD PTR [r15+0x8],rcx
   1dd49:	cmp    rax,rcx
   1dd4c:	je     1deb0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x7a0>
   1dd52:	sbb    r9d,r9d
   1dd55:	or     r9d,0x1
   1dd59:	cmp    r14d,0x8
   1dd5d:	jg     1df72 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x862>
   1dd63:	test   rbx,rbx
   1dd66:	je     1dda0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x690>
   1dd68:	mov    rcx,QWORD PTR [rbx+0x130]
   1dd6f:	xor    edx,edx
   1dd71:	mov    r14d,0x4
   1dd77:	mov    esi,0x5
   1dd7c:	mov    eax,0x140
   1dd81:	jmp    1d83d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x12d>
   1dd86:	cmp    esi,0x8
   1dd89:	jg     1daf7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x3e7>
   1dd8f:	mov    rbx,rbp
   1dd92:	mov    r9d,0x1
   1dd98:	jmp    1dd59 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x649>
   1dd9a:	nop    WORD PTR [rax+rax*1+0x0]
   1dda0:	xor    edx,edx
   1dda2:	mov    r14d,0x4
   1dda8:	mov    esi,0x5
   1ddad:	xor    ecx,ecx
   1ddaf:	mov    eax,0x140
   1ddb4:	jmp    1d83d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x12d>
   1ddb9:	nop    DWORD PTR [rax+0x0]
   1ddc0:	lea    rdi,[rsp+0x3c]
   1ddc5:	mov    DWORD PTR [rsp],edx
   1ddc8:	call   174e0 <_ZN11nodeversionI22nodeversion_parametersIjEE6unlockES2_>
   1ddcd:	mov    eax,DWORD PTR [rbx+0x18]
   1ddd0:	mov    esi,DWORD PTR [rsp+0x3c]
   1ddd4:	mov    edx,DWORD PTR [rsp]
   1ddd7:	and    eax,0xf
   1ddda:	jmp    1db9e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x48e>
   1dddf:	nop
   1dde0:	mov    esi,0x5
   1dde5:	mov    rdi,r13
   1dde8:	mov    QWORD PTR [rsp],rax
   1ddec:	call   59690 <_ZN10threadinfo11refill_poolEi>
   1ddf1:	mov    rbx,QWORD PTR [r13+0x60]
   1ddf5:	mov    rax,QWORD PTR [rsp]
   1ddf9:	test   rbx,rbx
   1ddfc:	je     1db2a <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x41a>
   1de02:	jmp    1db23 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x413>
   1de07:	nop    WORD PTR [rax+rax*1+0x0]
   1de10:	mov    rbp,rbx
   1de13:	jmp    1db85 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x475>
   1de18:	nop    DWORD PTR [rax+rax*1+0x0]
   1de20:	mov    rax,QWORD PTR [rcx+0x8]
   1de24:	jmp    1dbfb <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x4eb>
   1de29:	nop    DWORD PTR [rax+0x0]
   1de30:	mov    rcx,QWORD PTR [rcx+0x8]
   1de34:	bswap  rcx
   1de37:	mov    QWORD PTR [r15+0x8],rcx
   1de3b:	cmp    rcx,rax
   1de3e:	jne    1dc54 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x544>
   1de44:	cmp    r14d,0x8
   1de48:	jle    1de80 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x770>
   1de4a:	cmp    esi,0x8
   1de4d:	je     1de9e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x78e>
   1de4f:	mov    r14,rbx
   1de52:	jmp    1db08 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x3f8>
   1de57:	nop    WORD PTR [rax+rax*1+0x0]
   1de60:	lea    rdi,[r15+0x50]
   1de64:	xor    esi,esi
   1de66:	mov    DWORD PTR [rsp],edx
   1de69:	call   1ba90 <_ZN12small_vectorISt4pairIPN8Masstree4leafINS1_26default_query_table_paramsEEEjELj1ESaIS6_EE4growEj>
   1de6e:	mov    rax,QWORD PTR [r15+0x58]
   1de72:	mov    edx,DWORD PTR [rsp]
   1de75:	jmp    1dbb3 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x4a3>
   1de7a:	nop    WORD PTR [rax+rax*1+0x0]
   1de80:	sub    edx,edi
   1de82:	mov    r9d,edx
   1de85:	je     1de4f <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x73f>
   1de87:	cmp    esi,0x8
   1de8a:	jg     1d7f0 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0xe0>
   1de90:	jmp    1dd63 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x653>
   1de95:	nop    DWORD PTR [rax]
   1de98:	cmp    r14d,0x8
   1de9c:	jle    1de80 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x770>
   1de9e:	mov    r9d,0x1
   1dea4:	xor    esi,esi
   1dea6:	jmp    1d7f3 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0xe3>
   1deab:	nop    DWORD PTR [rax+rax*1+0x0]
   1deb0:	cmp    r14d,0x8
   1deb4:	jg     1de9e <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x78e>
   1deb6:	sub    edx,edi
   1deb8:	mov    r9d,edx
   1debb:	jmp    1dd63 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x653>
   1dec0:	lea    rcx,[rip+0x3cd79]        # 5ac40 <_IO_stdin_used+0xc40>
   1dec7:	mov    edx,0x20e
   1decc:	lea    rsi,[rip+0x3e09c]        # 5bf6f <_IO_stdin_used+0x1f6f>
   1ded3:	lea    rdi,[rip+0x3e68c]        # 5c566 <_IO_stdin_used+0x2566>
   1deda:	call   5960 <__assert_fail@plt>
   1dedf:	mov    esi,0x40
   1dee4:	sub    esi,edx
   1dee6:	movsxd rdx,esi
   1dee9:	add    rax,rdx
   1deec:	lea    r14,[rax+0x3f]
   1def0:	shr    r14,0x6
   1def4:	mov    esi,r14d
   1def7:	sub    r14d,0x1
   1defb:	test   al,0x3f
   1defd:	setne  dl
   1df00:	cmp    rax,0x1fff
   1df06:	seta   dil
   1df0a:	or     edx,edi
   1df0c:	jmp    1d83d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x12d>
   1df11:	mov    rax,QWORD PTR [rsp+0x68]
   1df16:	sub    rax,QWORD PTR fs:0x28
   1df1f:	jne    1df6d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x85d>
   1df21:	xor    ecx,ecx
   1df23:	lea    rdx,[rip+0x3e64c]        # 5c576 <_IO_stdin_used+0x2576>
   1df2a:	mov    esi,0x121
   1df2f:	lea    rdi,[rip+0x3e039]        # 5bf6f <_IO_stdin_used+0x1f6f>
   1df36:	call   590f0 <_Z26fail_masstree_preconditionPKciS0_S0_>
   1df3b:	mov    rbx,rbp
   1df3e:	jmp    1d7e7 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0xd7>
   1df43:	mov    rax,QWORD PTR [rsp+0x68]
   1df48:	sub    rax,QWORD PTR fs:0x28
   1df51:	jne    1df6d <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0x85d>
   1df53:	xor    ecx,ecx
   1df55:	lea    rdx,[rip+0x3e5c6]        # 5c522 <_IO_stdin_used+0x2522>
   1df5c:	mov    esi,0x90
   1df61:	lea    rdi,[rip+0x3dfa2]        # 5bf0a <_IO_stdin_used+0x1f0a>
   1df68:	call   59090 <_Z23fail_masstree_invariantPKciS0_S0_>
   1df6d:	call   5ac0 <__stack_chk_fail@plt>
   1df72:	xor    esi,esi
   1df74:	jmp    1d7f3 <_ZN8Masstree7tcursorINS_26default_query_table_paramsEE14make_new_layerER10threadinfo+0xe3>
   1df79:	nop    DWORD PTR [rax+0x0]

000000000001df80 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo>:
   1df80:	endbr64
   1df84:	push   r15
   1df86:	movsxd r15,esi
   1df89:	push   r14
   1df8b:	push   r13
   1df8d:	push   r12
   1df8f:	push   rbp
   1df90:	push   rbx
   1df91:	mov    rbx,rdi
   1df94:	sub    rsp,0x38
   1df98:	mov    QWORD PTR [rdi+r15*8+0x98],0x0
   1dfa4:	mov    rax,QWORD PTR [rdx]
   1dfa7:	mov    QWORD PTR [rsp+0x10],rcx
   1dfac:	mov    QWORD PTR [rdi+r15*8+0x20],rax
   1dfb1:	mov    eax,DWORD PTR [rdx+0x8]
   1dfb4:	cmp    eax,0x8
   1dfb7:	jg     1dfd0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x50>
   1dfb9:	mov    BYTE PTR [rdi+r15*1+0x6],al
   1dfbe:	add    rsp,0x38
   1dfc2:	pop    rbx
   1dfc3:	pop    rbp
   1dfc4:	pop    r12
   1dfc6:	pop    r13
   1dfc8:	pop    r14
   1dfca:	pop    r15
   1dfcc:	ret
   1dfcd:	nop    DWORD PTR [rax]
   1dfd0:	mov    BYTE PTR [rdi+r15*1+0x6],0x40
   1dfd6:	mov    eax,DWORD PTR [rdx+0x8]
   1dfd9:	mov    rbp,r15
   1dfdc:	mov    r14,QWORD PTR [rdx+0x10]
   1dfe0:	mov    rdi,QWORD PTR [rdi+0x110]
   1dfe7:	sub    eax,0x8
   1dfea:	mov    DWORD PTR [rsp+0xc],eax
   1dfee:	lea    rax,[r14+0x8]
   1dff2:	mov    QWORD PTR [rsp+0x18],rax
   1dff7:	test   rdi,rdi
   1dffa:	je     1e00f <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x8f>
   1dffc:	mov    ecx,DWORD PTR [rsp+0xc]
   1e000:	mov    rdx,rax
   1e003:	mov    esi,r15d
   1e006:	call   1d140 <_ZN9stringbagItE6assignEiPKci>
   1e00b:	test   al,al
   1e00d:	jne    1dfbe <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x3e>
   1e00f:	cmp    BYTE PTR [rbx+0x4],0x0
   1e013:	mov    edx,DWORD PTR [rsp+0xc]
   1e017:	jg     1e260 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x2e0>
   1e01d:	mov    r13,QWORD PTR [rbx+0x18]
   1e021:	mov    r14,QWORD PTR [rbx+0x110]
   1e028:	mov    r15d,r13d
   1e02b:	and    r15d,0xf
   1e02f:	je     1e096 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x116>
   1e031:	lea    esi,[r15*4+0x4]
   1e039:	mov    ecx,0x4
   1e03e:	xor    edx,edx
   1e040:	test   r14,r14
   1e043:	je     1e1db <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x25b>
   1e049:	mov    rax,r13
   1e04c:	shr    rax,cl
   1e04f:	and    eax,0xf
   1e052:	cmp    ebp,eax
   1e054:	jne    1e074 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0xf4>
   1e056:	cs nop WORD PTR [rax+rax*1+0x0]
   1e060:	add    ecx,0x4
   1e063:	cmp    esi,ecx
   1e065:	je     1e090 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x110>
   1e067:	mov    rax,r13
   1e06a:	shr    rax,cl
   1e06d:	and    eax,0xf
   1e070:	cmp    ebp,eax
   1e072:	je     1e060 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0xe0>
   1e074:	cmp    BYTE PTR [rbx+rax*1+0x6],0x40
   1e079:	jne    1e060 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0xe0>
   1e07b:	movzx  eax,WORD PTR [r14+rax*4+0x6]
   1e081:	add    ecx,0x4
   1e084:	add    rdx,rax
   1e087:	cmp    esi,ecx
   1e089:	jne    1e067 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0xe7>
   1e08b:	nop    DWORD PTR [rax+rax*1+0x0]
   1e090:	mov    eax,DWORD PTR [rsp+0xc]
   1e094:	add    edx,eax
   1e096:	lea    rax,[rdx+0x47]
   1e09a:	add    rdx,0x46
   1e09e:	and    rdx,rax
   1e0a1:	bsr    rcx,rax
   1e0a5:	mov    eax,0x1
   1e0aa:	cmp    rdx,0x1
   1e0ae:	sbb    ecx,0xffffffff
   1e0b1:	shl    rax,cl
   1e0b4:	mov    rdx,rax
   1e0b7:	test   r14,r14
   1e0ba:	je     1e0cc <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x14c>
   1e0bc:	movzx  eax,WORD PTR [r14+0x2]
   1e0c1:	add    rax,0x1
   1e0c5:	cmp    rdx,rax
   1e0c8:	cmovb  rdx,rax
   1e0cc:	mov    rdi,rdx
   1e0cf:	mov    QWORD PTR [rsp+0x20],rdx
   1e0d4:	call   5ba0 <malloc@plt>
   1e0d9:	mov    rdx,QWORD PTR [rsp+0x20]
   1e0de:	mov    r12,rax
   1e0e1:	cmp    rdx,0x10000
   1e0e8:	ja     1e36e <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x3ee>
   1e0ee:	mov    eax,0x40
   1e0f3:	pxor   xmm0,xmm0
   1e0f7:	mov    QWORD PTR [r12+0x34],0x0
   1e100:	mov    WORD PTR [r12],ax
   1e105:	lea    eax,[rdx-0x1]
   1e108:	mov    WORD PTR [r12+0x2],ax
   1e10e:	mov    DWORD PTR [r12+0x3c],0x0
   1e117:	movups XMMWORD PTR [r12+0x4],xmm0
   1e11d:	movups XMMWORD PTR [r12+0x14],xmm0
   1e123:	movups XMMWORD PTR [r12+0x24],xmm0
   1e129:	test   r15d,r15d
   1e12c:	je     1e1f8 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x278>
   1e132:	test   r14,r14
   1e135:	je     1e2b0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x330>
   1e13b:	lea    r9d,[r15*4+0x4]
   1e143:	mov    r15d,0x4
   1e149:	jmp    1e15d <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x1dd>
   1e14b:	nop    DWORD PTR [rax+rax*1+0x0]
   1e150:	add    r15d,0x4
   1e154:	cmp    r15d,r9d
   1e157:	je     1e1f8 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x278>
   1e15d:	mov    rsi,r13
   1e160:	mov    ecx,r15d
   1e163:	shr    rsi,cl
   1e166:	and    esi,0xf
   1e169:	cmp    ebp,esi
   1e16b:	je     1e150 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x1d0>
   1e16d:	movsxd rax,esi
   1e170:	cmp    BYTE PTR [rbx+rax*1+0x6],0x40
   1e175:	jne    1e150 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x1d0>
   1e177:	movzx  edx,WORD PTR [r14+rax*4+0x4]
   1e17d:	movzx  ecx,WORD PTR [r14+rax*4+0x6]
   1e183:	mov    rdi,r12
   1e186:	mov    DWORD PTR [rsp+0x20],r9d
   1e18b:	add    rdx,r14
   1e18e:	call   1d140 <_ZN9stringbagItE6assignEiPKci>
   1e193:	mov    r9d,DWORD PTR [rsp+0x20]
   1e198:	test   al,al
   1e19a:	jne    1e150 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x1d0>
   1e19c:	lea    rcx,[rip+0x3c6c5]        # 5a868 <_IO_stdin_used+0x868>
   1e1a3:	mov    edx,0x2f3
   1e1a8:	lea    rsi,[rip+0x3ddc0]        # 5bf6f <_IO_stdin_used+0x1f6f>
   1e1af:	lea    rdi,[rip+0x3e14c]        # 5c302 <_IO_stdin_used+0x2302>
   1e1b6:	call   5960 <__assert_fail@plt>
   1e1bb:	nop    DWORD PTR [rax+rax*1+0x0]
   1e1c0:	movzx  eax,BYTE PTR [rbx+rax*2+0x13b]
   1e1c8:	add    rdx,rax
   1e1cb:	nop    DWORD PTR [rax+rax*1+0x0]
   1e1d0:	add    ecx,0x4
   1e1d3:	cmp    ecx,esi
   1e1d5:	je     1e090 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x110>
   1e1db:	mov    rax,r13
   1e1de:	shr    rax,cl
   1e1e1:	and    eax,0xf
   1e1e4:	cmp    ebp,eax
   1e1e6:	je     1e1d0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x250>
   1e1e8:	cmp    BYTE PTR [rbx+rax*1+0x6],0x40
   1e1ed:	jne    1e1d0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x250>
   1e1ef:	jmp    1e1c0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x240>
   1e1f1:	nop    DWORD PTR [rax+0x0]
   1e1f8:	mov    ecx,DWORD PTR [rsp+0xc]
   1e1fc:	mov    rdx,QWORD PTR [rsp+0x18]
   1e201:	mov    esi,ebp
   1e203:	mov    rdi,r12
   1e206:	call   1d140 <_ZN9stringbagItE6assignEiPKci>
   1e20b:	test   al,al
   1e20d:	je     1e3a7 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x427>
   1e213:	cmp    BYTE PTR [rbx+0x5],0x1
   1e217:	je     1e38d <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x40d>
   1e21d:	mov    QWORD PTR [rbx+0x110],r12
   1e224:	movzx  eax,BYTE PTR [rbx+0x4]
   1e228:	test   al,al
   1e22a:	js     1e231 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x2b1>
   1e22c:	not    eax
   1e22e:	mov    BYTE PTR [rbx+0x4],al
   1e231:	test   r14,r14
   1e234:	je     1dfbe <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x3e>
   1e23a:	mov    rdi,QWORD PTR [rsp+0x10]
   1e23f:	add    rsp,0x38
   1e243:	mov    rsi,r14
   1e246:	mov    edx,0x1200
   1e24b:	pop    rbx
   1e24c:	pop    rbp
   1e24d:	pop    r12
   1e24f:	pop    r13
   1e251:	pop    r14
   1e253:	pop    r15
   1e255:	jmp    b7c0 <_ZN10threadinfo14deallocate_rcuEPvm6memtag.isra.0>
   1e25a:	nop    WORD PTR [rax+rax*1+0x0]
   1e260:	lea    r15,[rbx+r15*2]
   1e264:	movzx  eax,BYTE PTR [r15+0x13b]
   1e26c:	cmp    eax,edx
   1e26e:	jb     1e330 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x3b0>
   1e274:	movzx  eax,BYTE PTR [r15+0x13a]
   1e27c:	mov    ebp,edx
   1e27e:	mov    r12d,eax
   1e281:	mov    eax,eax
   1e283:	movsxd rdx,DWORD PTR [rsp+0xc]
   1e288:	mov    rsi,QWORD PTR [rsp+0x18]
   1e28d:	lea    rdi,[rbx+rax*1+0x138]
   1e295:	call   59e0 <memcpy@plt>
   1e29a:	mov    BYTE PTR [r15+0x13a],r12b
   1e2a1:	mov    BYTE PTR [r15+0x13b],bpl
   1e2a8:	jmp    1dfbe <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x3e>
   1e2ad:	nop    DWORD PTR [rax]
   1e2b0:	lea    rax,[rbx+0x138]
   1e2b7:	lea    r11d,[r15*4+0x4]
   1e2bf:	mov    r15d,0x4
   1e2c5:	mov    QWORD PTR [rsp+0x28],rax
   1e2ca:	jmp    1e2dd <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x35d>
   1e2cc:	nop    DWORD PTR [rax+0x0]
   1e2d0:	add    r15d,0x4
   1e2d4:	cmp    r15d,r11d
   1e2d7:	je     1e1f8 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x278>
   1e2dd:	mov    rsi,r13
   1e2e0:	mov    ecx,r15d
   1e2e3:	shr    rsi,cl
   1e2e6:	and    esi,0xf
   1e2e9:	cmp    ebp,esi
   1e2eb:	je     1e2d0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x350>
   1e2ed:	movsxd rax,esi
   1e2f0:	cmp    BYTE PTR [rbx+rax*1+0x6],0x40
   1e2f5:	jne    1e2d0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x350>
   1e2f7:	movzx  edx,BYTE PTR [rbx+rax*2+0x13a]
   1e2ff:	movzx  ecx,BYTE PTR [rbx+rax*2+0x13b]
   1e307:	mov    rdi,r12
   1e30a:	mov    DWORD PTR [rsp+0x20],r11d
   1e30f:	mov    rax,QWORD PTR [rsp+0x28]
   1e314:	add    rdx,rax
   1e317:	call   1d140 <_ZN9stringbagItE6assignEiPKci>
   1e31c:	mov    r11d,DWORD PTR [rsp+0x20]
   1e321:	test   al,al
   1e323:	jne    1e2d0 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x350>
   1e325:	jmp    1e19c <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x21c>
   1e32a:	nop    WORD PTR [rax+rax*1+0x0]
   1e330:	mov    esi,0x8
   1e335:	movzx  eax,BYTE PTR [rbx+0x138]
   1e33c:	movzx  edi,BYTE PTR [rbx+0x139]
   1e343:	cmp    edx,esi
   1e345:	cmovge esi,edx
   1e348:	add    rdi,0x1
   1e34c:	mov    r12d,eax
   1e34f:	add    esi,eax
   1e351:	mov    esi,esi
   1e353:	cmp    rdi,rsi
   1e356:	jb     1e01d <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x9d>
   1e35c:	mov    edi,edx
   1e35e:	mov    ebp,edx
   1e360:	lea    edx,[rax+rdi*1]
   1e363:	mov    BYTE PTR [rbx+0x138],dl
   1e369:	jmp    1e281 <_ZN8Masstree4leafINS_26default_query_table_paramsEE6assignEiRKNS_3keyImEER10threadinfo+0x301>
   1e36e:	lea    rcx,[rip+0x3c463]        # 5a7d8 <_IO_stdin_used+0x7d8>
   1e375:	mov    edx,0x5e
   1e37a:	lea    rsi,[rip+0x3dd98]        # 5c119 <_IO_stdin_used+0x2119>
   1e381:	lea    rdi,[rip+0x3c4b0]        # 5a838 <_IO_stdin_used+0x838>
   1e388:	call   5960 <__assert_fail@plt>
   1e38d:	xor    ecx,ecx
   1e38f:	lea    rdx,[rip+0x3dd90]        # 5c126 <_IO_stdin_used+0x2126>
   1e396:	mov    esi,0x2ff
   1e39b:	lea    rdi,[rip+0x3dbcd]        # 5bf6f <_IO_stdin_used+0x1f6f>
   1e3a2:	call   59090 <_Z23fail_masstree_invariantPKciS0_S0_>
   1e3a7:	lea    rcx,[rip+0x3c4ba]        # 5a868 <_IO_stdin_used+0x868>
   1e3ae:	mov    edx,0x2f7
   1e3b3:	lea    rsi,[rip+0x3dbb5]        # 5bf6f <_IO_stdin_used+0x1f6f>
   1e3ba:	lea    rdi,[rip+0x3df41]        # 5c302 <_IO_stdin_used+0x2302>
   1e3c1:	call   5960 <__assert_fail@plt>
   1e3c6:	cs nop WORD PTR [rax+rax*1+0x0]
