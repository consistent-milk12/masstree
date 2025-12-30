masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get:
.Lfunc_begin158:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception52
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
	sub rsp, 232
	.cfi_def_cfa_offset 288
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	.cfi_offset rbp, -16
	mov r14, rdx
	mov qword ptr [rsp + 24], rsi
.Ltmp16776:
	mov qword ptr [rsp + 16], rdi
.Ltmp16777:
	test byte ptr fs:[seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF], 1
	je .LBB158_262
.Ltmp16778:
	mov rax, qword ptr fs:[0]
	lea rax, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
.Ltmp16779:
	mov rcx, qword ptr [rax + 24]
	mov qword ptr [rsp + 160], rcx
	movups xmm0, xmmword ptr [rax + 8]
	movaps xmmword ptr [rsp + 144], xmm0
.Ltmp16780:
.LBB158_2:
	mov rbx, qword ptr [rsp + 152]
.Ltmp16781:
	mov rsi, qword ptr [rsp + 160]
.Ltmp16782:
	mov rax, qword ptr [rsp + 16]
.Ltmp16783:
	mov rax, qword ptr [rax + 8*rsi + 472]
.Ltmp16784:
	test rax, rax
	je .LBB158_263
.Ltmp16785:
.LBB158_3:
	shl rbx, 8
.Ltmp16786:
	lea r15, [rax + rbx]
.Ltmp16787:
	movzx eax, byte ptr [rax + rbx + 128]
.Ltmp16788:
	test al, al
	je .LBB158_264
.Ltmp16789:
.LBB158_4:
	mov rax, qword ptr [r15 + 8]
.Ltmp16790:
	lea rcx, [rax + 1]
.Ltmp16791:
	mov qword ptr [r15 + 8], rcx
.Ltmp16792:
	test rax, rax
	jne .LBB158_9
.Ltmp16793:
	movzx eax, byte ptr [rip + seize::raw::membarrier::linux::STRATEGY.0]
.Ltmp16794:
	cmp al, 2
	jne .LBB158_7
.Ltmp16795:
	xor eax, eax
.Ltmp16796:
	xchg qword ptr [r15], rax
.Ltmp16797:
	jmp .LBB158_8
.Ltmp16798:
.LBB158_7:
	mov qword ptr [r15], 0
.Ltmp16799:
.LBB158_8:
	#MEMBARRIER
.LBB158_9:
	cmp r14, 257
	jae .LBB158_265
.Ltmp16800:
	cmp r14, 7
	mov qword ptr [rsp + 8], r15
.Ltmp16801:
	mov qword ptr [rsp + 64], r14
.Ltmp16802:
	ja .LBB158_14
.Ltmp16803:
	test r14, r14
.Ltmp16804:
	jne .LBB158_267
.Ltmp16805:
	xor r13d, r13d
.Ltmp16806:
.LBB158_13:
	mov rax, qword ptr [rsp + 16]
.Ltmp16807:
	mov rbx, qword ptr [rax + 1024]
	jmp .LBB158_15
.Ltmp16808:
.LBB158_14:
	mov rax, qword ptr [rsp + 24]
.Ltmp16809:
	mov r13, qword ptr [rax]
.Ltmp16810:
	bswap r13
.Ltmp16811:
	mov rax, qword ptr [rsp + 16]
.Ltmp16812:
	mov rbx, qword ptr [rax + 1024]
.Ltmp16813:
	cmp r14, 8
	jne .LBB158_119
.Ltmp16814:
.LBB158_15:
	mov qword ptr [rsp + 32], 0
.LBB158_16:
	mov r15, rbx
	lea rax, [rbx + 264]
	mov qword ptr [rsp + 48], rax
	lea rax, [rbx + 560]
	mov qword ptr [rsp + 40], rax
.Ltmp16816:
.LBB158_17:
	mov eax, dword ptr [r15]
.Ltmp16817:
	test eax, eax
	mov rax, qword ptr [rsp + 48]
.Ltmp16818:
	cmovs rax, qword ptr [rsp + 40]
	mov rbx, qword ptr [rax]
.Ltmp16819:
	mov rcx, r15
.Ltmp16820:
	test rbx, rbx
.Ltmp16821:
	jne .LBB158_16
.Ltmp16822:
	.p2align	4
.LBB158_18:
	mov r12, rcx
.Ltmp16824:
	mov ebx, dword ptr [rcx]
	test bl, 6
	je .LBB158_24
	xor eax, eax
	jmp .LBB158_21
	.p2align	4
.LBB158_20:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ebx, dword ptr [r12]
	test bl, 6
	je .LBB158_24
.LBB158_21:
	xor ecx, ecx
	.p2align	4
.LBB158_22:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB158_20
	cmp ecx, eax
	jbe .LBB158_22
	jmp .LBB158_20
.Ltmp16830:
	.p2align	4
.LBB158_24:
	#MEMBARRIER
	mov eax, dword ptr [r12]
.Ltmp16832:
	test eax, eax
.Ltmp16833:
	js .LBB158_107
.Ltmp16834:
.Ltmp16764:
	mov rdi, r13
	mov rsi, r12
	call masstree::ksearch::upper_bound_internode_generic
.Ltmp16765:
.Ltmp16835:
	cmp eax, 15
	jne .LBB158_28
.Ltmp16836:
	mov rax, qword ptr [r12 + 256]
.Ltmp16837:
	mov rcx, r15
.Ltmp16838:
	test rax, rax
.Ltmp16839:
	jne .LBB158_29
	jmp .LBB158_18
.Ltmp16840:
	.p2align	4
.LBB158_28:
	mov rax, qword ptr [r12 + 8*rax + 136]
.Ltmp16842:
	mov rcx, r15
.Ltmp16843:
	test rax, rax
.Ltmp16844:
	je .LBB158_18
.Ltmp16845:
.LBB158_29:
	prefetcht0 byte ptr [rax]
.Ltmp16846:
	#MEMBARRIER
	mov edx, dword ptr [r12]
.Ltmp16847:
	xor edx, ebx
	mov rcx, rax
	cmp edx, 4
.Ltmp16848:
	jb .LBB158_18
.Ltmp16849:
	#MEMBARRIER
	mov eax, dword ptr [r12]
.Ltmp16850:
	xor eax, ebx
	cmp eax, 512
.Ltmp16851:
	cmovae r12, r15
.Ltmp16852:
	mov rcx, r12
.Ltmp16853:
	jmp .LBB158_18
.Ltmp16854:
	.p2align	4
.LBB158_31:
	lea r14, [rbp + 64]
	jmp .LBB158_34
.Ltmp16856:
	.p2align	4
.LBB158_32:
	#MEMBARRIER
.LBB158_33:
	mov ecx, dword ptr [rbp]
	mov ebx, eax
.Ltmp16857:
	test ecx, 536870912
.Ltmp16858:
	jne .LBB158_17
.Ltmp16859:
.LBB158_34:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp16860:
.Ltmp16767:
	mov rdi, r14
	call rax
.Ltmp16861:
.Ltmp16768:
	mov rcx, rax
	and rcx, 31
.Ltmp16862:
	mov r10, qword ptr [rsp + 64]
.Ltmp16863:
	je .LBB158_43
.Ltmp16864:
	lea rsi, [rcx + 4*rcx]
	xor r8d, r8d
	jmp .LBB158_38
.Ltmp16865:
	.p2align	4
.LBB158_37:
	add r8, 5
.Ltmp16867:
	cmp rsi, r8
.Ltmp16868:
	je .LBB158_43
.Ltmp16869:
.LBB158_38:
	lea ecx, [r8 + 5]
	mov r9, rax
	shrd r9, rdx, cl
	mov rdi, rdx
	shr rdi, cl
	test cl, 64
	cmove rdi, r9
	and edi, 31
.Ltmp16870:
	cmp rdi, 23
	ja .LBB158_261
.Ltmp16871:
	mov rcx, qword ptr [rbp + 8*rdi + 128]
.Ltmp16872:
	cmp rcx, r13
	jne .LBB158_37
.Ltmp16873:
	movzx r9d, byte ptr [rbp + rdi + 320]
.Ltmp16874:
	mov rcx, qword ptr [rbp + 8*rdi + 344]
.Ltmp16875:
	test rcx, rcx
.Ltmp16876:
	je .LBB158_37
	cmp r9b, r10b
	jne .LBB158_37
.Ltmp16878:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp16879:
	xor eax, ebx
	cmp eax, 3
.Ltmp16880:
	ja .LBB158_44
	jmp .LBB158_114
.Ltmp16881:
	.p2align	4
.LBB158_43:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp16883:
	xor eax, ebx
	cmp eax, 3
.Ltmp16884:
	jbe .LBB158_97
.Ltmp16885:
.LBB158_44:
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_50
	xor ecx, ecx
	jmp .LBB158_47
.Ltmp16887:
	.p2align	4
.LBB158_46:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_50
.LBB158_47:
	xor eax, eax
	.p2align	4
.LBB158_48:
	mov edx, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp edx, ecx
	jae .LBB158_46
	cmp eax, ecx
	jbe .LBB158_48
	jmp .LBB158_46
.Ltmp16891:
	.p2align	4
.LBB158_50:
	#MEMBARRIER
	#MEMBARRIER
	mov ecx, dword ptr [rbp]
.Ltmp16893:
	xor ecx, ebx
	cmp ecx, 511
.Ltmp16894:
	ja .LBB158_52
.Ltmp16895:
	mov ecx, dword ptr [rbp]
.Ltmp16896:
	test cl, 4
.Ltmp16897:
	je .LBB158_33
.LBB158_52:
	mov eax, dword ptr [rbp]
.Ltmp16899:
	test al, 6
	je .LBB158_58
	xor ecx, ecx
	jmp .LBB158_55
	.p2align	4
.LBB158_54:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_58
.LBB158_55:
	xor eax, eax
	.p2align	4
.LBB158_56:
	mov edx, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp edx, ecx
	jae .LBB158_54
	cmp eax, ecx
	jbe .LBB158_56
	jmp .LBB158_54
.Ltmp16905:
	.p2align	4
.LBB158_58:
	#MEMBARRIER
	mov ecx, dword ptr [rbp]
.Ltmp16907:
	test ecx, 536870912
.Ltmp16908:
	jne .LBB158_33
.Ltmp16909:
	mov r12, rbp
	jmp .LBB158_61
.Ltmp16910:
	.p2align	4
.LBB158_60:
	mov rcx, r12
.Ltmp16911:
	mov edx, dword ptr [rcx]
	mov r12, rcx
.Ltmp16912:
	test edx, 536870912
.Ltmp16913:
	jne .LBB158_96
.Ltmp16914:
.LBB158_61:
	mov rcx, qword ptr [r12 + 544]
.Ltmp16915:
	test cl, 1
.Ltmp16916:
	jne .LBB158_70
.Ltmp16917:
	test rcx, rcx
.Ltmp16918:
	je .LBB158_96
.Ltmp16919:
	mov rdx, qword ptr [rcx + 128]
.Ltmp16920:
	cmp r13, rdx
	jb .LBB158_96
.Ltmp16921:
	mov eax, dword ptr [rcx]
.Ltmp16922:
	test al, 6
	je .LBB158_95
	xor edx, edx
.Ltmp16924:
	jmp .LBB158_67
	.p2align	4
.LBB158_66:
	and edx, 7
	lea edx, [2*rdx + 1]
	mov eax, dword ptr [rcx]
	test al, 6
	je .LBB158_95
.LBB158_67:
	xor eax, eax
	.p2align	4
.LBB158_68:
	mov esi, eax
	pause
	cmp eax, edx
	adc eax, 0
	cmp esi, edx
	jae .LBB158_66
	cmp eax, edx
	jbe .LBB158_68
	jmp .LBB158_66
.Ltmp16929:
	.p2align	4
.LBB158_70:
	xor ecx, ecx
.Ltmp16931:
	jmp .LBB158_72
.Ltmp16932:
	.p2align	4
.LBB158_71:
	#MEMBARRIER
	inc rcx
	cmp rcx, 1001
	je .LBB158_60
.Ltmp16934:
.LBB158_72:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16935:
	test dl, 1
.Ltmp16936:
	je .LBB158_60
.Ltmp16937:
	pause
.Ltmp16938:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16939:
	test dl, 1
	je .LBB158_60
.Ltmp16940:
	pause
.Ltmp16941:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16942:
	test dl, 1
	je .LBB158_60
.Ltmp16943:
	pause
.Ltmp16944:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16945:
	test dl, 1
	je .LBB158_60
.Ltmp16946:
	pause
.Ltmp16947:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16948:
	test dl, 1
	je .LBB158_60
.Ltmp16949:
	pause
.Ltmp16950:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16951:
	test dl, 1
	je .LBB158_60
.Ltmp16952:
	pause
.Ltmp16953:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16954:
	test dl, 1
	je .LBB158_60
.Ltmp16955:
	pause
.Ltmp16956:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16957:
	test dl, 1
	je .LBB158_60
.Ltmp16958:
	pause
.Ltmp16959:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16960:
	test dl, 1
	je .LBB158_60
.Ltmp16961:
	pause
.Ltmp16962:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16963:
	test dl, 1
	je .LBB158_60
.Ltmp16964:
	pause
.Ltmp16965:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16966:
	test dl, 1
	je .LBB158_60
.Ltmp16967:
	pause
.Ltmp16968:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16969:
	test dl, 1
	je .LBB158_60
.Ltmp16970:
	pause
.Ltmp16971:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16972:
	test dl, 1
	je .LBB158_60
.Ltmp16973:
	pause
.Ltmp16974:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16975:
	test dl, 1
	je .LBB158_60
.Ltmp16976:
	pause
.Ltmp16977:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16978:
	test dl, 1
	je .LBB158_60
.Ltmp16979:
	pause
.Ltmp16980:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16981:
	test dl, 1
	je .LBB158_60
.Ltmp16982:
	pause
.Ltmp16983:
	mov rdx, qword ptr [r12 + 544]
.Ltmp16984:
	test dl, 1
	je .LBB158_60
.Ltmp16985:
	mov edx, dword ptr [r12]
	test dl, 6
	je .LBB158_71
.Ltmp16986:
	xor edx, edx
	jmp .LBB158_92
	.p2align	4
.LBB158_91:
	and edx, 7
	lea edx, [2*rdx + 1]
	mov esi, dword ptr [r12]
	test sil, 6
	je .LBB158_71
.LBB158_92:
	xor esi, esi
	.p2align	4
.LBB158_93:
	mov edi, esi
	pause
	cmp esi, edx
	adc esi, 0
	cmp edi, edx
	jae .LBB158_91
	cmp esi, edx
	jbe .LBB158_93
	jmp .LBB158_91
.Ltmp16991:
	.p2align	4
.LBB158_95:
	#MEMBARRIER
	mov edx, dword ptr [rcx]
	mov r12, rcx
.Ltmp16993:
	test edx, 536870912
.Ltmp16994:
	je .LBB158_61
.Ltmp16995:
.LBB158_96:
	cmp r12, rbp
.Ltmp16996:
	je .LBB158_33
	jmp .LBB158_107
.Ltmp16997:
	.p2align	4
.LBB158_97:
	mov eax, dword ptr [rbp]
.Ltmp16999:
	test al, 6
.Ltmp17000:
	je .LBB158_104
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_32
	xor ecx, ecx
	jmp .LBB158_101
.Ltmp17003:
	.p2align	4
.LBB158_100:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_32
.LBB158_101:
	xor eax, eax
	.p2align	4
.LBB158_102:
	mov edx, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp edx, ecx
	jae .LBB158_100
	cmp eax, ecx
	jbe .LBB158_102
	jmp .LBB158_100
.Ltmp17007:
	.p2align	4
.LBB158_104:
	mov rax, qword ptr [rbp + 544]
.Ltmp17009:
	mov ecx, eax
	and ecx, 1
	neg rcx
.Ltmp17010:
	jb .LBB158_116
.Ltmp17011:
	lea r12, [rax + rcx]
.Ltmp17012:
	test r12, r12
	je .LBB158_116
.Ltmp17013:
	mov rax, qword ptr [r12 + 128]
.Ltmp17014:
	cmp r13, rax
	jb .LBB158_116
.Ltmp17015:
	.p2align	4
.LBB158_107:
	mov rbp, r12
.Ltmp17017:
	mov ebx, dword ptr [r12]
	test bl, 6
	je .LBB158_113
	xor eax, eax
	jmp .LBB158_110
	.p2align	4
.LBB158_109:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ebx, dword ptr [rbp]
	test bl, 6
	je .LBB158_113
.LBB158_110:
	xor ecx, ecx
	.p2align	4
.LBB158_111:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB158_109
	cmp ecx, eax
	jbe .LBB158_111
	jmp .LBB158_109
.Ltmp17023:
	.p2align	4
.LBB158_113:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp17025:
	test eax, 536870912
.Ltmp17026:
	je .LBB158_31
	jmp .LBB158_17
.Ltmp17027:
.LBB158_114:
	lock inc	qword ptr [rcx - 16]
.Ltmp17028:
	jle .LBB158_266
.Ltmp17029:
	add rcx, -16
.Ltmp17030:
	mov qword ptr [rsp + 32], rcx
.Ltmp17031:
.LBB158_116:
	mov rdx, qword ptr [rsp + 8]
.Ltmp17032:
	mov rax, qword ptr [rdx + 8]
.Ltmp17033:
	lea rcx, [rax - 1]
.Ltmp17034:
	mov qword ptr [rdx + 8], rcx
.Ltmp17035:
	cmp rax, 1
	jne .LBB158_118
	mov rsi, -1
.Ltmp17037:
	xchg qword ptr [rdx], rsi
.Ltmp17038:
	cmp rsi, -1
	jne .LBB158_269
.Ltmp17039:
.LBB158_118:
	mov rax, qword ptr [rsp + 32]
	add rsp, 232
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
	ret
.LBB158_119:
	.cfi_def_cfa_offset 288
	mov eax, 8
	xor r12d, r12d
.LBB158_122:
	lea rcx, [8*r12]
	mov rsi, r14
	sub rsi, rcx
	mov edx, 0
	cmovb rsi, rdx
	cmp rsi, 9
	mov edi, 64
	mov qword ptr [rsp + 40], rsi
	cmovb edi, esi
	sub r14, rax
	mov ecx, 0
	mov qword ptr [rsp + 32], rcx
	cmovb r14, rdx
	mov qword ptr [rsp + 56], r14
	mov rcx, qword ptr [rsp + 24]
	lea rcx, [rcx + rax]
	mov eax, 1
	cmovbe rcx, rax
	mov qword ptr [rsp + 104], rcx
	mov qword ptr [rsp + 72], r12
	mov dword ptr [rsp + 48], edi
.LBB158_123:
	lea rax, [rbx + 264]
	mov qword ptr [rsp + 88], rax
	mov qword ptr [rsp + 96], rbx
	lea rax, [rbx + 560]
	mov qword ptr [rsp + 80], rax
.Ltmp17043:
	.p2align	4
.LBB158_124:
	mov rbp, qword ptr [rsp + 96]
.Ltmp17044:
	mov eax, dword ptr [rbp]
.Ltmp17045:
	test eax, eax
	mov rax, qword ptr [rsp + 88]
.Ltmp17046:
	cmovs rax, qword ptr [rsp + 80]
	mov rbx, qword ptr [rax]
.Ltmp17047:
	mov rcx, rbp
.Ltmp17048:
	test rbx, rbx
	mov r15, qword ptr [rsp + 8]
.Ltmp17049:
	jne .LBB158_123
.Ltmp17050:
	.p2align	4
.LBB158_125:
	mov r14, rcx
.Ltmp17052:
	mov ebx, dword ptr [rcx]
	test bl, 6
	je .LBB158_131
	xor eax, eax
	jmp .LBB158_128
	.p2align	4
.LBB158_127:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ebx, dword ptr [r14]
	test bl, 6
	je .LBB158_131
.LBB158_128:
	xor ecx, ecx
	.p2align	4
.LBB158_129:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB158_127
	cmp ecx, eax
	jbe .LBB158_129
	jmp .LBB158_127
.Ltmp17058:
	.p2align	4
.LBB158_131:
	#MEMBARRIER
	mov eax, dword ptr [r14]
.Ltmp17060:
	test eax, eax
.Ltmp17061:
	js .LBB158_243
.Ltmp17062:
.Ltmp16751:
	mov rdi, r13
	mov rsi, r14
	call masstree::ksearch::upper_bound_internode_generic
.Ltmp16752:
.Ltmp17063:
	cmp eax, 15
	jne .LBB158_135
.Ltmp17064:
	mov rax, qword ptr [r14 + 256]
.Ltmp17065:
	mov rcx, rbp
.Ltmp17066:
	test rax, rax
.Ltmp17067:
	jne .LBB158_136
	jmp .LBB158_125
.Ltmp17068:
	.p2align	4
.LBB158_135:
	mov rax, qword ptr [r14 + 8*rax + 136]
.Ltmp17070:
	mov rcx, rbp
.Ltmp17071:
	test rax, rax
.Ltmp17072:
	je .LBB158_125
.Ltmp17073:
.LBB158_136:
	prefetcht0 byte ptr [rax]
.Ltmp17074:
	#MEMBARRIER
	mov edx, dword ptr [r14]
.Ltmp17075:
	xor edx, ebx
	mov rcx, rax
	cmp edx, 4
.Ltmp17076:
	jb .LBB158_125
.Ltmp17077:
	#MEMBARRIER
	mov eax, dword ptr [r14]
.Ltmp17078:
	xor eax, ebx
	cmp eax, 512
.Ltmp17079:
	cmovae r14, rbp
.Ltmp17080:
	mov rcx, r14
.Ltmp17081:
	jmp .LBB158_125
.Ltmp17082:
	.p2align	4
.LBB158_138:
	lea rax, [rbp + 64]
	mov qword ptr [rsp + 128], rax
	jmp .LBB158_141
.Ltmp17084:
	.p2align	4
.LBB158_139:
	#MEMBARRIER
.LBB158_140:
	mov ecx, dword ptr [rbp]
	mov r14d, eax
.Ltmp17085:
	test ecx, 536870912
.Ltmp17086:
	jne .LBB158_124
.Ltmp17087:
.LBB158_141:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp16754:
	mov rdi, qword ptr [rsp + 128]
.Ltmp17089:
	call rax
.Ltmp17090:
.Ltmp16755:
	mov rcx, rax
	and rcx, 31
.Ltmp17091:
	je .LBB158_159
.Ltmp17092:
	lea r15, [rcx + 4*rcx]
	xor r12d, r12d
	jmp .LBB158_145
.Ltmp17093:
	.p2align	4
.LBB158_144:
	add r12, 5
.Ltmp17095:
	cmp r15, r12
.Ltmp17096:
	je .LBB158_159
.Ltmp17097:
.LBB158_145:
	lea ecx, [r12 + 5]
	mov rsi, rax
	shrd rsi, rdx, cl
	mov rdi, rdx
	shr rdi, cl
	test cl, 64
	cmove rdi, rsi
	and edi, 31
.Ltmp17098:
	cmp rdi, 23
	ja .LBB158_261
.Ltmp17099:
	mov rcx, qword ptr [rbp + 8*rdi + 128]
.Ltmp17100:
	cmp rcx, r13
	jne .LBB158_144
.Ltmp17101:
	movzx ecx, byte ptr [rbp + rdi + 320]
.Ltmp17102:
	mov rbx, qword ptr [rbp + 8*rdi + 344]
.Ltmp17103:
	test rbx, rbx
.Ltmp17104:
	je .LBB158_144
	mov esi, dword ptr [rsp + 48]
	cmp cl, sil
	jne .LBB158_156
	cmp sil, 64
	jne .LBB158_238
.Ltmp17107:
	movzx ecx, byte ptr [rbp + rdi + 320]
.Ltmp17108:
	cmp cl, 64
.Ltmp17109:
	jne .LBB158_144
.Ltmp17110:
	mov rcx, qword ptr [rbp + 536]
.Ltmp17111:
	test rcx, rcx
.Ltmp17112:
	je .LBB158_144
.Ltmp17113:
	mov r9d, dword ptr [rcx + 8*rdi + 24]
	mov esi, 4294967295
	cmp r9, rsi
	je .LBB158_144
	movzx edi, word ptr [rcx + 8*rdi + 28]
.Ltmp17116:
	lea rsi, [rdi + r9]
.Ltmp17117:
	mov r8, qword ptr [rcx + 16]
.Ltmp17118:
	cmp rsi, r8
.Ltmp17119:
	ja .LBB158_258
.Ltmp17120:
	cmp qword ptr [rsp + 56], rdi
	jne .LBB158_144
.Ltmp17121:
	add r9, qword ptr [rcx + 8]
.Ltmp17122:
	mov rdi, r9
.Ltmp17123:
	mov rsi, qword ptr [rsp + 104]
.Ltmp17124:
	mov qword ptr [rsp + 112], rdx
	mov rdx, qword ptr [rsp + 56]
	mov qword ptr [rsp + 120], rax
	call qword ptr [rip + bcmp@GOTPCREL]
.Ltmp17125:
	mov rdx, qword ptr [rsp + 112]
	mov ecx, eax
	mov rax, qword ptr [rsp + 120]
	test ecx, ecx
.Ltmp17126:
	jne .LBB158_144
	jmp .LBB158_239
.Ltmp17127:
.LBB158_156:
	cmp qword ptr [rsp + 40], 9
	jb .LBB158_144
	test cl, cl
	jns .LBB158_144
.Ltmp17129:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp17130:
	xor eax, r14d
	cmp eax, 3
.Ltmp17131:
	ja .LBB158_160
	jmp .LBB158_250
.Ltmp17132:
	.p2align	4
.LBB158_159:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp17134:
	xor eax, r14d
	cmp eax, 3
.Ltmp17135:
	jbe .LBB158_230
.Ltmp17136:
.LBB158_160:
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_166
	xor ecx, ecx
	jmp .LBB158_163
	.p2align	4
.LBB158_162:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_166
.LBB158_163:
	xor eax, eax
	.p2align	4
.LBB158_164:
	mov edx, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp edx, ecx
	jae .LBB158_162
	cmp eax, ecx
	jbe .LBB158_164
	jmp .LBB158_162
.Ltmp17142:
	.p2align	4
.LBB158_166:
	#MEMBARRIER
	#MEMBARRIER
	mov ecx, dword ptr [rbp]
.Ltmp17144:
	xor ecx, r14d
	cmp ecx, 511
.Ltmp17145:
	ja .LBB158_168
.Ltmp17146:
	mov ecx, dword ptr [rbp]
.Ltmp17147:
	test cl, 4
.Ltmp17148:
	je .LBB158_140
.LBB158_168:
	mov eax, dword ptr [rbp]
.Ltmp17150:
	test al, 6
	je .LBB158_174
	xor ecx, ecx
	jmp .LBB158_171
	.p2align	4
.LBB158_170:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_174
.LBB158_171:
	xor eax, eax
	.p2align	4
.LBB158_172:
	mov edx, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp edx, ecx
	jae .LBB158_170
	cmp eax, ecx
	jbe .LBB158_172
	jmp .LBB158_170
.Ltmp17156:
	.p2align	4
.LBB158_174:
	#MEMBARRIER
	mov ecx, dword ptr [rbp]
.Ltmp17158:
	test ecx, 536870912
.Ltmp17159:
	jne .LBB158_140
.Ltmp17160:
	mov r14, rbp
.Ltmp17161:
	jmp .LBB158_178
.Ltmp17162:
	.p2align	4
.LBB158_212:
	mov rcx, r14
.Ltmp17163:
	mov edx, dword ptr [rcx]
	mov r14, rcx
.Ltmp17164:
	test edx, 536870912
.Ltmp17165:
	je .LBB158_178
	jmp .LBB158_237
.Ltmp17166:
	.p2align	4
.LBB158_176:
	#MEMBARRIER
	mov edx, dword ptr [rcx]
	mov r14, rcx
.Ltmp17168:
	test edx, 536870912
.Ltmp17169:
	jne .LBB158_237
.Ltmp17170:
.LBB158_178:
	mov rcx, qword ptr [r14 + 544]
.Ltmp17171:
	test cl, 1
.Ltmp17172:
	jne .LBB158_187
.Ltmp17173:
	test rcx, rcx
.Ltmp17174:
	je .LBB158_237
.Ltmp17175:
	mov rdx, qword ptr [rcx + 128]
.Ltmp17176:
	cmp r13, rdx
	jb .LBB158_237
.Ltmp17177:
	mov eax, dword ptr [rcx]
.Ltmp17178:
	test al, 6
	je .LBB158_176
	xor edx, edx
.Ltmp17180:
	jmp .LBB158_184
	.p2align	4
.LBB158_183:
	and edx, 7
	lea edx, [2*rdx + 1]
	mov eax, dword ptr [rcx]
	test al, 6
	je .LBB158_176
.LBB158_184:
	xor eax, eax
	.p2align	4
.LBB158_185:
	mov esi, eax
	pause
	cmp eax, edx
	adc eax, 0
	cmp esi, edx
	jae .LBB158_183
	cmp eax, edx
	jbe .LBB158_185
	jmp .LBB158_183
.Ltmp17185:
	.p2align	4
.LBB158_187:
	xor ecx, ecx
.Ltmp17187:
	jmp .LBB158_189
.Ltmp17188:
	.p2align	4
.LBB158_188:
	#MEMBARRIER
	inc rcx
	cmp rcx, 1001
	je .LBB158_212
.Ltmp17190:
.LBB158_189:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17191:
	test dl, 1
.Ltmp17192:
	je .LBB158_212
.Ltmp17193:
	pause
.Ltmp17194:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17195:
	test dl, 1
.Ltmp17196:
	je .LBB158_212
.Ltmp17197:
	pause
.Ltmp17198:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17199:
	test dl, 1
.Ltmp17200:
	je .LBB158_212
.Ltmp17201:
	pause
.Ltmp17202:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17203:
	test dl, 1
.Ltmp17204:
	je .LBB158_212
.Ltmp17205:
	pause
.Ltmp17206:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17207:
	test dl, 1
.Ltmp17208:
	je .LBB158_212
.Ltmp17209:
	pause
.Ltmp17210:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17211:
	test dl, 1
.Ltmp17212:
	je .LBB158_212
.Ltmp17213:
	pause
.Ltmp17214:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17215:
	test dl, 1
.Ltmp17216:
	je .LBB158_212
.Ltmp17217:
	pause
.Ltmp17218:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17219:
	test dl, 1
.Ltmp17220:
	je .LBB158_212
.Ltmp17221:
	pause
.Ltmp17222:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17223:
	test dl, 1
.Ltmp17224:
	je .LBB158_212
.Ltmp17225:
	pause
.Ltmp17226:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17227:
	test dl, 1
.Ltmp17228:
	je .LBB158_212
.Ltmp17229:
	pause
.Ltmp17230:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17231:
	test dl, 1
.Ltmp17232:
	je .LBB158_212
.Ltmp17233:
	pause
.Ltmp17234:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17235:
	test dl, 1
.Ltmp17236:
	je .LBB158_212
.Ltmp17237:
	pause
.Ltmp17238:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17239:
	test dl, 1
.Ltmp17240:
	je .LBB158_212
.Ltmp17241:
	pause
.Ltmp17242:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17243:
	test dl, 1
.Ltmp17244:
	je .LBB158_212
.Ltmp17245:
	pause
.Ltmp17246:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17247:
	test dl, 1
.Ltmp17248:
	je .LBB158_212
.Ltmp17249:
	pause
.Ltmp17250:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17251:
	test dl, 1
.Ltmp17252:
	je .LBB158_212
.Ltmp17253:
	pause
.Ltmp17254:
	mov rdx, qword ptr [r14 + 544]
.Ltmp17255:
	test dl, 1
.Ltmp17256:
	je .LBB158_212
.Ltmp17257:
	mov edx, dword ptr [r14]
	test dl, 6
	je .LBB158_188
.Ltmp17258:
	xor edx, edx
	jmp .LBB158_209
	.p2align	4
.LBB158_208:
	and edx, 7
	lea edx, [2*rdx + 1]
	mov esi, dword ptr [r14]
	test sil, 6
	je .LBB158_188
.LBB158_209:
	xor esi, esi
	.p2align	4
.LBB158_210:
	mov edi, esi
	pause
	cmp esi, edx
	adc esi, 0
	cmp edi, edx
	jae .LBB158_208
	cmp esi, edx
	jbe .LBB158_210
	jmp .LBB158_208
.Ltmp17263:
	.p2align	4
.LBB158_230:
	mov eax, dword ptr [rbp]
.Ltmp17265:
	test al, 6
.Ltmp17266:
	je .LBB158_240
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_139
	xor ecx, ecx
	jmp .LBB158_234
	.p2align	4
.LBB158_233:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov eax, dword ptr [rbp]
	test al, 6
	je .LBB158_139
.LBB158_234:
	xor eax, eax
	.p2align	4
.LBB158_235:
	mov edx, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp edx, ecx
	jae .LBB158_233
	cmp eax, ecx
	jbe .LBB158_235
	jmp .LBB158_233
.Ltmp17273:
	.p2align	4
.LBB158_237:
	cmp r14, rbp
.Ltmp17275:
	je .LBB158_140
	jmp .LBB158_243
.Ltmp17276:
.LBB158_238:
	mov rax, qword ptr [rsp + 40]
	mov ecx, eax
.Ltmp17277:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp17278:
	xor eax, r14d
	cmp eax, 3
.Ltmp17279:
	ja .LBB158_160
	jmp .LBB158_250
.Ltmp17280:
.LBB158_239:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp17281:
	xor eax, r14d
	cmp eax, 4
.Ltmp17282:
	jae .LBB158_160
	jmp .LBB158_259
.Ltmp17283:
	.p2align	4
.LBB158_240:
	mov rax, qword ptr [rbp + 544]
.Ltmp17285:
	mov ecx, eax
	and ecx, 1
	neg rcx
.Ltmp17286:
	jb .LBB158_116
.Ltmp17287:
	lea r14, [rax + rcx]
.Ltmp17288:
	test r14, r14
	je .LBB158_116
.Ltmp17289:
	mov rax, qword ptr [r14 + 128]
.Ltmp17290:
	cmp r13, rax
	jb .LBB158_116
.Ltmp17291:
.LBB158_243:
	mov rbp, r14
.Ltmp17292:
	mov r14d, dword ptr [r14]
.Ltmp17293:
	test r14b, 6
	je .LBB158_249
	xor eax, eax
	jmp .LBB158_246
	.p2align	4
.LBB158_245:
	and eax, 7
	lea eax, [2*rax + 1]
	mov r14d, dword ptr [rbp]
	test r14b, 6
	je .LBB158_249
.LBB158_246:
	xor ecx, ecx
	.p2align	4
.LBB158_247:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB158_245
	cmp ecx, eax
	jbe .LBB158_247
	jmp .LBB158_245
.Ltmp17299:
	.p2align	4
.LBB158_249:
	#MEMBARRIER
	mov eax, dword ptr [rbp]
.Ltmp17301:
	test eax, 536870912
.Ltmp17302:
	je .LBB158_138
	jmp .LBB158_124
.Ltmp17303:
	.p2align	4
.LBB158_250:
	test cl, cl
	jns .LBB158_259
	mov r12, qword ptr [rsp + 72]
.Ltmp17306:
	lea rax, [8*r12 + 8]
.Ltmp17307:
	mov r14, qword ptr [rsp + 64]
.Ltmp17308:
	mov rsi, r14
	sub rsi, rax
.Ltmp17309:
	jb .LBB158_120
.Ltmp17310:
	mov rcx, qword ptr [rsp + 24]
.Ltmp17311:
	lea rdi, [rcx + 8*r12]
	add rdi, 8
.Ltmp17312:
	cmp rsi, 8
	mov r15, qword ptr [rsp + 8]
.Ltmp17313:
	jae .LBB158_255
.Ltmp17314:
	cmp r14, rax
	jne .LBB158_256
.Ltmp17315:
.LBB158_120:
	xor r13d, r13d
.Ltmp17316:
	jmp .LBB158_121
.Ltmp17317:
.LBB158_255:
	mov r13, qword ptr [rdi]
.Ltmp17318:
	bswap r13
.Ltmp17319:
.LBB158_121:
	lea rax, [8*r12 + 16]
.Ltmp17320:
	inc r12
.Ltmp17321:
	cmp r14, rax
	cmovb rax, r14
.Ltmp17322:
	jmp .LBB158_122
.Ltmp17323:
.LBB158_256:
.Ltmp16757:
	call masstree::key::Key::read_ikey_slow
.Ltmp17324:
.Ltmp16758:
	mov r13, rax
.Ltmp17325:
	jmp .LBB158_121
.Ltmp17326:
.LBB158_258:
.Ltmp16760:
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.197]
.Ltmp17327:
	mov rdi, r9
	mov rdx, r8
	mov r15, qword ptr [rsp + 8]
	call core::slice::index::slice_index_fail
.Ltmp17328:
	jmp .LBB158_266
.Ltmp17329:
.LBB158_259:
	lock inc	qword ptr [rbx - 16]
.Ltmp17330:
	jle .LBB158_266
.Ltmp17331:
	add rbx, -16
.Ltmp17332:
	mov qword ptr [rsp + 32], rbx
.Ltmp17333:
	jmp .LBB158_116
.Ltmp17334:
.LBB158_261:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
	mov esi, 24
	mov r15, qword ptr [rsp + 8]
	call core::panicking::panic_bounds_check
.Ltmp16771:
	jmp .LBB158_266
.Ltmp17335:
.LBB158_262:
	mov rax, qword ptr fs:[0]
	lea rsi, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
	lea rdi, [rsp + 144]
	call seize::raw::tls::thread_id::Thread::init_slow
	jmp .LBB158_2
.Ltmp17336:
.LBB158_263:
	mov rax, qword ptr [rsp + 16]
.Ltmp17337:
	lea rdi, [rax + 8*rsi]
	add rdi, 472
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp17338:
	jmp .LBB158_3
.Ltmp17339:
.LBB158_264:
	mov rdi, r15
	call seize::raw::tls::ThreadLocal<T>::write
	jmp .LBB158_4
.Ltmp17340:
.LBB158_265:
	mov qword ptr [rsp + 136], r14
	lea rax, [rsp + 136]
	mov qword ptr [rsp + 200], rax
	lea rax, [rip + core::fmt::num::imp::<impl core::fmt::Display for u64>::fmt]
	mov qword ptr [rsp + 208], rax
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.156]
	mov qword ptr [rsp + 216], rcx
	mov qword ptr [rsp + 224], rax
.Ltmp17344:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.159]
.Ltmp17345:
	mov qword ptr [rsp + 144], rax
	mov qword ptr [rsp + 152], 2
	mov qword ptr [rsp + 176], 0
	lea rax, [rsp + 200]
.Ltmp17346:
	mov qword ptr [rsp + 160], rax
	mov qword ptr [rsp + 168], 2
.Ltmp17347:
.Ltmp16749:
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.161]
	lea rdi, [rsp + 144]
	call core::panicking::panic_fmt
.Ltmp17348:
.LBB158_266:
	ud2
.Ltmp17349:
.LBB158_267:
	mov rdi, qword ptr [rsp + 24]
.Ltmp17350:
	mov rsi, r14
	call masstree::key::Key::read_ikey_slow
.Ltmp17351:
.Ltmp16763:
	mov r13, rax
	jmp .LBB158_13
.Ltmp17352:
.LBB158_269:
	#MEMBARRIER
	mov rdi, qword ptr [rsp + 16]
.Ltmp17353:
	call seize::raw::collector::Collector::traverse
.Ltmp17354:
	jmp .LBB158_118
.Ltmp17355:
.Ltmp16759:
	jmp .LBB158_276
.Ltmp16769:
	mov r15, qword ptr [rsp + 8]
	jmp .LBB158_276
.Ltmp16766:
	mov r15, qword ptr [rsp + 8]
	jmp .LBB158_276
.Ltmp16756:
	mov r15, qword ptr [rsp + 8]
	jmp .LBB158_276
.Ltmp16753:
	jmp .LBB158_276
.Ltmp17360:
.Ltmp16772:
.LBB158_276:
	mov rcx, qword ptr [r15 + 8]
.Ltmp17361:
	lea rdx, [rcx - 1]
.Ltmp17362:
	mov qword ptr [r15 + 8], rdx
.Ltmp17363:
	cmp rcx, 1
	jne .LBB158_278
	mov rsi, -1
.Ltmp17365:
	xchg qword ptr [r15], rsi
.Ltmp17366:
	cmp rsi, -1
	jne .LBB158_279
.Ltmp17367:
.LBB158_278:
	mov rdi, rax
	call _Unwind_Resume@PLT
.Ltmp17368:
.LBB158_279:
	#MEMBARRIER
.Ltmp16773:
	mov rdi, qword ptr [rsp + 16]
	mov rbx, rax
	call seize::raw::collector::Collector::traverse
.Ltmp17369:
	mov rax, rbx
.Ltmp16774:
	jmp .LBB158_278
.Ltmp17370:
.Ltmp16775:
	call core::panicking::panic_in_cleanup
.Lfunc_end158:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get, .Lfunc_end158-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get","a",@progbits
	.p2align	2, 0x0
GCC_except_table158:
masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get:
.Lfunc_begin231:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception102
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
	sub rsp, 56
	.cfi_def_cfa_offset 112
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	.cfi_offset rbp, -16
	mov r15, rsi
.Ltmp28195:
	test byte ptr fs:[seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF], 1
	je .LBB231_104
.Ltmp28196:
	mov rax, qword ptr fs:[0]
	lea rax, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
.Ltmp28197:
	mov rcx, qword ptr [rax + 24]
	mov qword ptr [rsp + 48], rcx
	movups xmm0, xmmword ptr [rax + 8]
	movaps xmmword ptr [rsp + 32], xmm0
.Ltmp28198:
.LBB231_2:
	mov rbx, qword ptr [rsp + 40]
.Ltmp28199:
	mov rsi, qword ptr [rsp + 48]
.Ltmp28200:
	mov rax, qword ptr [rdi + 8*rsi + 472]
.Ltmp28201:
	test rax, rax
	je .LBB231_105
.Ltmp28202:
.LBB231_3:
	shl rbx, 8
.Ltmp28203:
	lea r14, [rax + rbx]
.Ltmp28204:
	movzx eax, byte ptr [rax + rbx + 128]
.Ltmp28205:
	test al, al
	je .LBB231_106
.Ltmp28206:
.LBB231_4:
	mov rax, qword ptr [r14 + 8]
.Ltmp28207:
	lea rcx, [rax + 1]
.Ltmp28208:
	mov qword ptr [r14 + 8], rcx
.Ltmp28209:
	test rax, rax
	jne .LBB231_9
.Ltmp28210:
	movzx eax, byte ptr [rip + seize::raw::membarrier::linux::STRATEGY.0]
.Ltmp28211:
	cmp al, 2
	jne .LBB231_7
.Ltmp28212:
	xor eax, eax
.Ltmp28213:
	xchg qword ptr [r14], rax
.Ltmp28214:
	jmp .LBB231_8
.Ltmp28215:
.LBB231_7:
	mov qword ptr [r14], 0
.Ltmp28216:
.LBB231_8:
	#MEMBARRIER
.LBB231_9:
	mov qword ptr [rsp + 8], r14
.Ltmp28217:
	bswap r15
.Ltmp28218:
	mov qword ptr [rsp], rdi
.Ltmp28219:
	mov rax, qword ptr [rdi + 1024]
	mov qword ptr [rsp + 16], 0
.Ltmp28220:
.LBB231_10:
	mov r13, rax
	lea rbx, [rax + 264]
	add rax, 560
	mov qword ptr [rsp + 24], rax
.Ltmp28221:
.LBB231_11:
	mov eax, dword ptr [r13]
.Ltmp28222:
	test eax, eax
.Ltmp28223:
	mov rax, rbx
	cmovs rax, qword ptr [rsp + 24]
	mov rax, qword ptr [rax]
.Ltmp28224:
	test rax, rax
.Ltmp28225:
	jne .LBB231_10
.Ltmp28226:
.Ltmp28183:
	mov rdi, r13
	mov rsi, r15
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::reach_leaf_concurrent_generic
.Ltmp28184:
.LBB231_13:
	mov r14, rax
.Ltmp28228:
	mov r12d, dword ptr [rax]
	test r12b, 6
	je .LBB231_19
	xor eax, eax
	jmp .LBB231_16
	.p2align	4
.LBB231_15:
	and eax, 7
	lea eax, [2*rax + 1]
	mov r12d, dword ptr [r14]
	test r12b, 6
	je .LBB231_19
.LBB231_16:
	xor ecx, ecx
	.p2align	4
.LBB231_17:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB231_15
	cmp ecx, eax
	jbe .LBB231_17
	jmp .LBB231_15
.Ltmp28234:
	.p2align	4
.LBB231_19:
	#MEMBARRIER
	mov eax, dword ptr [r14]
.Ltmp28236:
	test eax, 536870912
.Ltmp28237:
	jne .LBB231_11
.Ltmp28238:
	lea rbp, [r14 + 64]
	jmp .LBB231_23
.Ltmp28239:
	.p2align	4
.LBB231_21:
	#MEMBARRIER
.LBB231_22:
	mov eax, dword ptr [r14]
	mov r12d, ecx
.Ltmp28240:
	test eax, 536870912
.Ltmp28241:
	jne .LBB231_11
.Ltmp28242:
.LBB231_23:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp28243:
.Ltmp28186:
	mov rdi, rbp
	call rax
.Ltmp28244:
.Ltmp28187:
	mov rcx, rax
	and rcx, 31
.Ltmp28245:
	je .LBB231_32
.Ltmp28246:
	lea rsi, [rcx + 4*rcx]
	xor r8d, r8d
	jmp .LBB231_27
.Ltmp28247:
	.p2align	4
.LBB231_26:
	add r8, 5
.Ltmp28249:
	cmp rsi, r8
.Ltmp28250:
	je .LBB231_32
.Ltmp28251:
.LBB231_27:
	lea ecx, [r8 + 5]
	mov r9, rax
	shrd r9, rdx, cl
	mov rdi, rdx
	shr rdi, cl
	test cl, 64
	cmove rdi, r9
	and edi, 31
.Ltmp28252:
	cmp rdi, 24
	jae .LBB231_102
.Ltmp28253:
	mov rcx, qword ptr [r14 + 8*rdi + 128]
.Ltmp28254:
	cmp rcx, r15
	jne .LBB231_26
.Ltmp28255:
	movzx r9d, byte ptr [r14 + rdi + 320]
.Ltmp28256:
	mov rcx, qword ptr [r14 + 8*rdi + 344]
.Ltmp28257:
	test rcx, rcx
.Ltmp28258:
	je .LBB231_26
	cmp r9b, 8
	jne .LBB231_26
.Ltmp28260:
	#MEMBARRIER
	mov eax, dword ptr [r14]
.Ltmp28261:
	xor eax, r12d
	cmp eax, 3
.Ltmp28262:
	ja .LBB231_33
	jmp .LBB231_97
.Ltmp28263:
	.p2align	4
.LBB231_32:
	#MEMBARRIER
	mov eax, dword ptr [r14]
.Ltmp28265:
	xor eax, r12d
	cmp eax, 3
.Ltmp28266:
	jbe .LBB231_85
.Ltmp28267:
.LBB231_33:
	mov ecx, dword ptr [r14]
	test cl, 6
	je .LBB231_39
	xor eax, eax
	jmp .LBB231_36
.Ltmp28269:
	.p2align	4
.LBB231_35:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ecx, dword ptr [r14]
	test cl, 6
	je .LBB231_39
.LBB231_36:
	xor ecx, ecx
	.p2align	4
.LBB231_37:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB231_35
	cmp ecx, eax
	jbe .LBB231_37
	jmp .LBB231_35
.Ltmp28273:
	.p2align	4
.LBB231_39:
	#MEMBARRIER
	#MEMBARRIER
	mov eax, dword ptr [r14]
.Ltmp28275:
	xor eax, r12d
	cmp eax, 511
.Ltmp28276:
	ja .LBB231_41
.Ltmp28277:
	mov eax, dword ptr [r14]
.Ltmp28278:
	test al, 4
.Ltmp28279:
	je .LBB231_22
.LBB231_41:
	mov ecx, dword ptr [r14]
.Ltmp28281:
	test cl, 6
	je .LBB231_47
	xor eax, eax
	jmp .LBB231_44
	.p2align	4
.LBB231_43:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ecx, dword ptr [r14]
	test cl, 6
	je .LBB231_47
.LBB231_44:
	xor ecx, ecx
	.p2align	4
.LBB231_45:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB231_43
	cmp ecx, eax
	jbe .LBB231_45
	jmp .LBB231_43
.Ltmp28287:
	.p2align	4
.LBB231_47:
	#MEMBARRIER
	mov eax, dword ptr [r14]
.Ltmp28289:
	test eax, 536870912
	jne .LBB231_22
.Ltmp28290:
	mov rax, r14
	jmp .LBB231_59
.Ltmp28291:
	.p2align	4
.LBB231_49:
	mov rdx, rax
.Ltmp28292:
	mov esi, dword ptr [rdx]
	mov rax, rdx
.Ltmp28293:
	test esi, 536870912
.Ltmp28294:
	jne .LBB231_92
.Ltmp28295:
.LBB231_59:
	mov rdx, qword ptr [rax + 544]
.Ltmp28296:
	test dl, 1
.Ltmp28297:
	jne .LBB231_60
.Ltmp28298:
	test rdx, rdx
.Ltmp28299:
	je .LBB231_92
.Ltmp28300:
	mov rsi, qword ptr [rdx + 128]
.Ltmp28301:
	cmp r15, rsi
	jb .LBB231_92
.Ltmp28302:
	mov ecx, dword ptr [rdx]
.Ltmp28303:
	test cl, 6
	je .LBB231_58
	xor eax, eax
	jmp .LBB231_55
.Ltmp28305:
	.p2align	4
.LBB231_54:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ecx, dword ptr [rdx]
	test cl, 6
	je .LBB231_58
.LBB231_55:
	xor ecx, ecx
	.p2align	4
.LBB231_56:
	mov esi, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp esi, eax
	jae .LBB231_54
	cmp ecx, eax
	jbe .LBB231_56
	jmp .LBB231_54
.Ltmp28309:
	.p2align	4
.LBB231_60:
	xor edx, edx
.Ltmp28311:
	jmp .LBB231_62
.Ltmp28312:
	.p2align	4
.LBB231_61:
	#MEMBARRIER
	inc rdx
	cmp rdx, 1001
	je .LBB231_49
.Ltmp28314:
.LBB231_62:
	mov rsi, qword ptr [rax + 544]
.Ltmp28315:
	test sil, 1
.Ltmp28316:
	je .LBB231_49
.Ltmp28317:
	pause
.Ltmp28318:
	mov rsi, qword ptr [rax + 544]
.Ltmp28319:
	test sil, 1
.Ltmp28320:
	je .LBB231_49
.Ltmp28321:
	pause
.Ltmp28322:
	mov rsi, qword ptr [rax + 544]
.Ltmp28323:
	test sil, 1
.Ltmp28324:
	je .LBB231_49
.Ltmp28325:
	pause
.Ltmp28326:
	mov rsi, qword ptr [rax + 544]
.Ltmp28327:
	test sil, 1
.Ltmp28328:
	je .LBB231_49
.Ltmp28329:
	pause
.Ltmp28330:
	mov rsi, qword ptr [rax + 544]
.Ltmp28331:
	test sil, 1
.Ltmp28332:
	je .LBB231_49
.Ltmp28333:
	pause
.Ltmp28334:
	mov rsi, qword ptr [rax + 544]
.Ltmp28335:
	test sil, 1
.Ltmp28336:
	je .LBB231_49
.Ltmp28337:
	pause
.Ltmp28338:
	mov rsi, qword ptr [rax + 544]
.Ltmp28339:
	test sil, 1
.Ltmp28340:
	je .LBB231_49
.Ltmp28341:
	pause
.Ltmp28342:
	mov rsi, qword ptr [rax + 544]
.Ltmp28343:
	test sil, 1
.Ltmp28344:
	je .LBB231_49
.Ltmp28345:
	pause
.Ltmp28346:
	mov rsi, qword ptr [rax + 544]
.Ltmp28347:
	test sil, 1
.Ltmp28348:
	je .LBB231_49
.Ltmp28349:
	pause
.Ltmp28350:
	mov rsi, qword ptr [rax + 544]
.Ltmp28351:
	test sil, 1
.Ltmp28352:
	je .LBB231_49
.Ltmp28353:
	pause
.Ltmp28354:
	mov rsi, qword ptr [rax + 544]
.Ltmp28355:
	test sil, 1
.Ltmp28356:
	je .LBB231_49
.Ltmp28357:
	pause
.Ltmp28358:
	mov rsi, qword ptr [rax + 544]
.Ltmp28359:
	test sil, 1
.Ltmp28360:
	je .LBB231_49
.Ltmp28361:
	pause
.Ltmp28362:
	mov rsi, qword ptr [rax + 544]
.Ltmp28363:
	test sil, 1
.Ltmp28364:
	je .LBB231_49
.Ltmp28365:
	pause
.Ltmp28366:
	mov rsi, qword ptr [rax + 544]
.Ltmp28367:
	test sil, 1
.Ltmp28368:
	je .LBB231_49
.Ltmp28369:
	pause
.Ltmp28370:
	mov rsi, qword ptr [rax + 544]
.Ltmp28371:
	test sil, 1
.Ltmp28372:
	je .LBB231_49
.Ltmp28373:
	pause
.Ltmp28374:
	mov rsi, qword ptr [rax + 544]
.Ltmp28375:
	test sil, 1
.Ltmp28376:
	je .LBB231_49
.Ltmp28377:
	pause
.Ltmp28378:
	mov rsi, qword ptr [rax + 544]
.Ltmp28379:
	test sil, 1
.Ltmp28380:
	je .LBB231_49
.Ltmp28381:
	mov esi, dword ptr [rax]
	test sil, 6
	je .LBB231_61
.Ltmp28382:
	xor esi, esi
	jmp .LBB231_82
	.p2align	4
.LBB231_81:
	and esi, 7
	lea esi, [2*rsi + 1]
	mov edi, dword ptr [rax]
	test dil, 6
	je .LBB231_61
.LBB231_82:
	xor edi, edi
	.p2align	4
.LBB231_83:
	mov r8d, edi
	pause
	cmp edi, esi
	adc edi, 0
	cmp r8d, esi
	jae .LBB231_81
	cmp edi, esi
	jbe .LBB231_83
	jmp .LBB231_81
.Ltmp28387:
	.p2align	4
.LBB231_58:
	#MEMBARRIER
	mov esi, dword ptr [rdx]
	mov rax, rdx
.Ltmp28389:
	test esi, 536870912
.Ltmp28390:
	je .LBB231_59
.Ltmp28391:
.LBB231_92:
	cmp rax, r14
.Ltmp28392:
	je .LBB231_22
	jmp .LBB231_13
.Ltmp28393:
	.p2align	4
.LBB231_85:
	mov eax, dword ptr [r14]
.Ltmp28395:
	test al, 6
.Ltmp28396:
	je .LBB231_93
	mov ecx, dword ptr [r14]
	test cl, 6
	je .LBB231_21
	xor eax, eax
	jmp .LBB231_89
.Ltmp28399:
	.p2align	4
.LBB231_88:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ecx, dword ptr [r14]
	test cl, 6
	je .LBB231_21
.LBB231_89:
	xor ecx, ecx
	.p2align	4
.LBB231_90:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB231_88
	cmp ecx, eax
	jbe .LBB231_90
	jmp .LBB231_88
.Ltmp28403:
	.p2align	4
.LBB231_93:
	mov rax, qword ptr [r14 + 544]
.Ltmp28405:
	mov ecx, eax
	and ecx, 1
	neg rcx
.Ltmp28406:
	jb .LBB231_101
.Ltmp28407:
	add rax, rcx
.Ltmp28408:
	test rax, rax
	je .LBB231_98
.Ltmp28409:
	mov rcx, qword ptr [rax + 128]
.Ltmp28410:
	cmp r15, rcx
	jae .LBB231_13
.Ltmp28411:
.LBB231_101:
	jmp .LBB231_98
.Ltmp28412:
.LBB231_97:
	mov rbp, qword ptr [rcx]
	mov eax, 1
	mov qword ptr [rsp + 16], rax
.Ltmp28413:
.LBB231_98:
	mov rdx, qword ptr [rsp + 8]
.Ltmp28414:
	mov rax, qword ptr [rdx + 8]
.Ltmp28415:
	lea rcx, [rax - 1]
.Ltmp28416:
	mov qword ptr [rdx + 8], rcx
.Ltmp28417:
	cmp rax, 1
	jne .LBB231_100
	mov rsi, -1
.Ltmp28419:
	xchg qword ptr [rdx], rsi
.Ltmp28420:
	cmp rsi, -1
	jne .LBB231_107
.Ltmp28421:
.LBB231_100:
	mov rax, qword ptr [rsp + 16]
	mov rdx, rbp
	add rsp, 56
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
	ret
.Ltmp28422:
.LBB231_102:
	.cfi_def_cfa_offset 112
.Ltmp28189:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
.Ltmp28423:
	mov esi, 24
	call core::panicking::panic_bounds_check
.Ltmp28424:
.Ltmp28190:
	ud2
.Ltmp28425:
.LBB231_104:
	mov rax, qword ptr fs:[0]
	lea rsi, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
	mov rbx, rdi
.Ltmp28426:
	lea rdi, [rsp + 32]
	call seize::raw::tls::thread_id::Thread::init_slow
	mov rdi, rbx
	jmp .LBB231_2
.Ltmp28427:
.LBB231_105:
	mov r14, rdi
.Ltmp28428:
	lea rdi, [rdi + 8*rsi]
	add rdi, 472
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp28429:
	mov rdi, r14
.Ltmp28430:
	jmp .LBB231_3
.Ltmp28431:
.LBB231_106:
	mov rbx, rdi
.Ltmp28432:
	mov rdi, r14
	call seize::raw::tls::ThreadLocal<T>::write
	mov rdi, rbx
	jmp .LBB231_4
.Ltmp28433:
.LBB231_107:
	#MEMBARRIER
	mov rdi, qword ptr [rsp]
.Ltmp28434:
	call seize::raw::collector::Collector::traverse
.Ltmp28435:
	jmp .LBB231_100
.Ltmp28436:
.Ltmp28185:
	jmp .LBB231_111
.Ltmp28188:
	jmp .LBB231_111
.Ltmp28191:
.LBB231_111:
	mov r15, rax
	mov rdi, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 8]
.Ltmp28192:
	call core::ptr::drop_in_place<seize::guard::LocalGuard>
.Ltmp28193:
	mov rdi, r15
	call _Unwind_Resume@PLT
.Ltmp28194:
	call core::panicking::panic_in_cleanup
.Lfunc_end231:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get, .Lfunc_end231-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::get","a",@progbits
	.p2align	2, 0x0
