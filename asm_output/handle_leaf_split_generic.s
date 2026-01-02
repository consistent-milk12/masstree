masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic:
.Lfunc_begin170:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception55
	push rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov rbp, rsp
	.cfi_def_cfa_register rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	and rsp, -64
	sub rsp, 832
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	mov rbx, r9
	mov r12d, ecx
	mov r15, rdx
	mov r13, rsi
	mov qword ptr [rsp + 56], rdi
.Ltmp17983:
.Ltmp17901:
	lea rdi, [rsp + 192]
.Ltmp17984:
	mov rdx, r8
.Ltmp17985:
	call <masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point
.Ltmp17986:
.Ltmp17902:
	cmp dword ptr [rsp + 192], 1
	jne .LBB170_15
	mov rax, qword ptr [rsp + 200]
	mov qword ptr [rsp + 64], rax
.Ltmp17989:
	mov ecx, dword ptr [r13]
	mov dword ptr [rsp + 52], 0
.Ltmp17990:
	mov rax, qword ptr [r13 + 560]
.Ltmp17991:
	mov edx, 0
.Ltmp17992:
	test ecx, 1073741824
.Ltmp17993:
	je .LBB170_4
.Ltmp17994:
	test rax, rax
	sete al
.Ltmp17995:
	mov rcx, qword ptr [rsp + 56]
.Ltmp17996:
	mov rcx, qword ptr [rcx + 1024]
.Ltmp17997:
	cmp rcx, r13
	sete dl
.Ltmp17998:
	setne cl
.Ltmp17999:
	and cl, al
	mov dword ptr [rsp + 52], ecx
.Ltmp18000:
.LBB170_4:
	mov dword ptr [rsp + 92], edx
.Ltmp18001:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 229], xmm0
	movups xmmword ptr [rsp + 213], xmm0
	movups xmmword ptr [rsp + 197], xmm0
	mov qword ptr [rsp + 244], 0
	lea rdi, [rsp + 272]
	mov edx, 456
	xor esi, esi
	call qword ptr [rip + memset@GOTPCREL]
	mov dword ptr [rsp + 192], -2147483648
	mov byte ptr [rsp + 196], 0
	movabs rax, 2399420405009046
	mov qword ptr [rsp + 264], rax
	movabs rax, -4127003096615560480
	mov qword ptr [rsp + 256], rax
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 728], xmm0
	movups xmmword ptr [rsp + 744], xmm0
.Ltmp18002:
	mov edi, 576
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp18003:
	test rax, rax
	je .LBB170_124
.Ltmp18004:
	mov r14, rax
	lea rsi, [rsp + 192]
	mov edx, 576
	mov rdi, rax
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp18005:
	or dword ptr [r15], 4
.Ltmp18006:
	#MEMBARRIER
.Ltmp17904:
	lea rdi, [rsp + 192]
	mov qword ptr [rsp + 176], rdi
.Ltmp18007:
	mov rsi, r13
	mov rdx, qword ptr [rsp + 64]
	mov rcx, r14
	mov r8, rbx
	call <masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated
.Ltmp17905:
	mov r14, qword ptr [rsp + 192]
.Ltmp18009:
	mov rax, qword ptr [rsp + 200]
	mov qword ptr [rsp + 80], rax
.Ltmp18011:
	mov rdx, qword ptr [rsp + 56]
.Ltmp18012:
	lea rax, [rdx + 960]
	mov qword ptr [rsp + 64], rax
.Ltmp18013:
	mov cl, 1
.Ltmp18014:
	xor eax, eax
	lock cmpxchg	byte ptr [rdx + 960], cl
.Ltmp18015:
	jne .LBB170_125
.Ltmp18016:
.LBB170_7:
	mov rdx, qword ptr [rsp + 56]
.Ltmp18017:
	mov rbx, qword ptr [rdx + 984]
.Ltmp18018:
	cmp rbx, qword ptr [rdx + 968]
	jne .LBB170_9
.Ltmp18019:
.Ltmp17908:
	lea rdi, [rdx + 968]
.Ltmp18020:
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp18021:
	mov rdx, qword ptr [rsp + 56]
.Ltmp17909:
.Ltmp18022:
.LBB170_9:
	mov rax, qword ptr [rdx + 976]
.Ltmp18023:
	mov qword ptr [rax + 8*rbx], r14
.Ltmp18024:
	inc rbx
.Ltmp18025:
	mov qword ptr [rdx + 984], rbx
	xor ecx, ecx
.Ltmp18026:
	mov al, 1
	lock cmpxchg	byte ptr [rdx + 960], cl
.Ltmp18027:
	je .LBB170_12
.Ltmp17914:
	mov rdi, qword ptr [rsp + 64]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17915:
	jmp .LBB170_12
.Ltmp18029:
	.p2align	4
.LBB170_11:
	mov rdi, r13
	call masstree::leaf24::LeafNode24<S>::wait_for_split
.Ltmp18031:
.LBB170_12:
	mov rcx, qword ptr [r13 + 544]
.Ltmp18032:
	test cl, 1
	jne .LBB170_11
.Ltmp18033:
.LBB170_13:
	lea rdx, [rcx + 1]
.Ltmp18034:
	mov rax, rcx
	lock cmpxchg	qword ptr [r13 + 544], rdx
.Ltmp18035:
	je .LBB170_17
.Ltmp18036:
	pause
.Ltmp18037:
	mov rcx, qword ptr [r13 + 544]
.Ltmp18038:
	test cl, 1
	jne .LBB170_11
	jmp .LBB170_13
.Ltmp18039:
.LBB170_15:
	test r12b, 4
	jne .LBB170_76
	mov eax, r12d
	and eax, 2
	lea r12d, [r12 + 4*rax]
.Ltmp18041:
	and r12d, -268435464
	jmp .LBB170_77
.Ltmp18042:
.LBB170_17:
	mov qword ptr [r14 + 552], r13
.Ltmp18043:
	mov qword ptr [r14 + 544], rcx
.Ltmp18044:
	test rcx, rcx
.Ltmp18045:
	je .LBB170_19
.Ltmp18046:
	mov qword ptr [rcx + 552], r14
.Ltmp18047:
.LBB170_19:
	#MEMBARRIER
	mov qword ptr [r13 + 544], r14
	mov esi, dword ptr [rsp + 92]
	mov byte ptr [rsp + 38], sil
	mov eax, dword ptr [rsp + 52]
	mov byte ptr [rsp + 39], al
	mov rax, qword ptr [rsp + 56]
	lea rcx, [rax + 992]
	mov qword ptr [rsp + 24], rcx
	add rax, 1000
	mov qword ptr [rsp + 96], rax
	mov eax, 1
	mov edi, 560
	mov cl, 1
	jmp .LBB170_21
.Ltmp18049:
	.p2align	4
.LBB170_20:
	test dl, 1
	mov rax, r14
	cmovne rax, r15
.Ltmp18050:
	mov rdx, qword ptr [rsp + 40]
.Ltmp18051:
	mov rcx, qword ptr [rsp + 64]
.Ltmp18052:
	mov qword ptr [rdx + rcx], rax
.Ltmp18053:
	mov eax, dword ptr [rdx]
.Ltmp18054:
	add eax, 512
	and eax, -1342177792
.Ltmp18055:
	mov ecx, dword ptr [rsp + 76]
.Ltmp18056:
	add ecx, 512
	and ecx, -1342177792
.Ltmp18057:
	cmp qword ptr [rsp + 184], r15
.Ltmp18058:
	#MEMBARRIER
	mov dword ptr [rdx], eax
	mov rax, qword ptr [rsp + 104]
.Ltmp18059:
	mov dword ptr [rax], ecx
.Ltmp18060:
	sete byte ptr [rsp + 38]
	mov eax, dword ptr [rsp + 52]
	mov byte ptr [rsp + 39], al
.Ltmp18061:
	mov rdx, qword ptr [rsp + 168]
.Ltmp18062:
	lea rax, [rdx + 1]
	xor ecx, ecx
	mov r13, r15
	cmp rdx, 63
	mov esi, dword ptr [rsp + 92]
	mov edi, 560
	ja .LBB170_136
.Ltmp18063:
.LBB170_21:
	mov qword ptr [rsp + 40], r14
.Ltmp18064:
	mov dword ptr [rsp + 76], r12d
.Ltmp18065:
	test cl, 1
	mov edx, 264
	cmovne rdx, rdi
	mov qword ptr [rsp + 64], rdx
	mov qword ptr [rsp + 104], r15
.Ltmp18066:
	not sil
	mov rdi, rax
	xor edx, edx
	mov r8d, esi
	jmp .LBB170_25
.Ltmp18067:
	.p2align	4
.LBB170_22:
	mov rax, qword ptr [rsp + 64]
.Ltmp18068:
	mov rax, qword ptr [r13 + rax]
.Ltmp18069:
	cmp rax, r15
	je .LBB170_36
	xor edx, edx
.Ltmp18071:
.LBB170_24:
	lea eax, [r12 + 512]
	mov esi, r12d
	add esi, 8
	test r12b, 4
	mov edi, -1342177792
	mov r9d, -268435464
	cmove edi, r9d
	cmovne esi, eax
	and esi, edi
	mov dword ptr [r15], esi
.Ltmp18072:
	pause
.Ltmp18073:
	mov rax, qword ptr [rsp + 168]
.Ltmp18074:
	cmp rax, 63
	lea rdi, [rax + 1]
.Ltmp18075:
	ja .LBB170_135
.Ltmp18076:
.LBB170_25:
	mov rax, qword ptr [rsp + 64]
.Ltmp18077:
	mov r15, qword ptr [r13 + rax]
.Ltmp18078:
	test r15, r15
.Ltmp18079:
	jne .LBB170_27
.Ltmp18080:
	test byte ptr [rsp + 52], 1
	jne .LBB170_78
.Ltmp18081:
.LBB170_27:
	test r15, r15
	setne al
.Ltmp18082:
	or al, r8b
	test al, 1
	je .LBB170_85
.Ltmp18083:
	test r15, r15
.Ltmp18084:
	je .LBB170_134
.Ltmp18085:
	mov qword ptr [rsp + 168], rdi
	xor esi, esi
	jmp .LBB170_31
.Ltmp18086:
	.p2align	4
.LBB170_30:
	and esi, 7
	lea esi, [2*rsi + 1]
.LBB170_31:
	mov r12d, dword ptr [r15]
	test r12b, 1
	jne .LBB170_33
	mov r14d, r12d
	or r14d, 3
	mov eax, r12d
	lock cmpxchg	dword ptr [r15], r14d
	je .LBB170_22
.LBB170_33:
	xor eax, eax
	.p2align	4
.LBB170_34:
	mov edi, eax
	pause
	cmp eax, esi
	adc eax, 0
	cmp edi, esi
	jae .LBB170_30
	cmp eax, esi
	jbe .LBB170_34
	jmp .LBB170_30
.Ltmp18092:
	.p2align	4
.LBB170_36:
	movzx eax, byte ptr [r15 + 4]
.Ltmp18094:
	movzx ebx, al
.Ltmp18095:
	test bl, bl
.Ltmp18096:
	je .LBB170_43
	xor eax, eax
	jmp .LBB170_38
.Ltmp18098:
	.p2align	4
.LBB170_40:
	mov rsi, qword ptr [r15 + 256]
.Ltmp18099:
	cmp rsi, r13
.Ltmp18100:
	je .LBB170_46
.Ltmp18101:
.LBB170_41:
	inc rax
.Ltmp18102:
	cmp rbx, rax
.Ltmp18103:
	je .LBB170_42
.Ltmp18104:
.LBB170_38:
	cmp rax, 15
	jae .LBB170_40
.Ltmp18105:
	mov rsi, qword ptr [r15 + 8*rax + 136]
.Ltmp18106:
	cmp rsi, r13
.Ltmp18107:
	jne .LBB170_41
	jmp .LBB170_46
.Ltmp18108:
.LBB170_42:
	dec rax
.Ltmp18109:
	cmp rax, 14
	jae .LBB170_44
.Ltmp18110:
.LBB170_43:
	mov rax, qword ptr [r15 + 8*rbx + 136]
.Ltmp18111:
	cmp rax, r13
.Ltmp18112:
	jne .LBB170_45
	jmp .LBB170_47
.Ltmp18113:
.LBB170_44:
	mov rax, qword ptr [r15 + 256]
.Ltmp18114:
	cmp rax, r13
.Ltmp18115:
	je .LBB170_47
.Ltmp18116:
.LBB170_45:
	inc rdx
	cmp rdx, 16
	jbe .LBB170_24
	jmp .LBB170_110
	.p2align	4
.LBB170_46:
	mov rbx, rax
.Ltmp18120:
.LBB170_47:
	movzx eax, byte ptr [r15 + 4]
.Ltmp18121:
	cmp al, 14
.Ltmp18122:
	jbe .LBB170_111
.Ltmp18123:
	or dword ptr [r15], 4
.Ltmp18124:
	#MEMBARRIER
	mov rax, qword ptr [rsp + 56]
.Ltmp18125:
	mov rdx, qword ptr [rax + 1024]
.Ltmp18126:
	cmp rdx, r15
	sete cl
	mov dword ptr [rsp + 92], ecx
.Ltmp18127:
	mov qword ptr [rsp + 184], rdx
.Ltmp18128:
	je .LBB170_51
.Ltmp18129:
	mov rax, qword ptr [r15 + 264]
.Ltmp18130:
	test rax, rax
.Ltmp18131:
	je .LBB170_52
.LBB170_51:
	mov dword ptr [rsp + 52], 0
	jmp .LBB170_53
.Ltmp18133:
.LBB170_52:
	mov eax, dword ptr [r15]
.Ltmp18134:
	shr eax, 30
.Ltmp18135:
	and al, 1
	mov dword ptr [rsp + 52], eax
.Ltmp18136:
.LBB170_53:
	or r12d, 7
.Ltmp18137:
	mov eax, dword ptr [r15 + 8]
.Ltmp18138:
	mov dword ptr [rsp + 132], eax
.Ltmp18139:
	mov r13d, dword ptr [r15]
.Ltmp18140:
	lea rax, [rsp + 196]
	xorps xmm0, xmm0
.Ltmp18141:
	movups xmmword ptr [rax + 96], xmm0
	movups xmmword ptr [rax + 80], xmm0
	movups xmmword ptr [rax + 64], xmm0
	movups xmmword ptr [rax + 48], xmm0
	movups xmmword ptr [rax + 32], xmm0
	movups xmmword ptr [rax + 16], xmm0
	movups xmmword ptr [rax], xmm0
	mov qword ptr [rax + 112], 0
.Ltmp18142:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp18143:
	test rax, rax
	je .LBB170_123
.Ltmp18144:
	mov r14, rax
	and r13d, -2147483648
	or r13d, 5
.Ltmp18145:
	mov dword ptr [rax], r13d
	mov byte ptr [rax + 4], 0
	mov eax, dword ptr [rsp + 132]
	mov dword ptr [r14 + 8], eax
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [r14 + 12], xmm0
	movups xmmword ptr [r14 + 28], xmm1
	movups xmmword ptr [r14 + 44], xmm2
	movups xmmword ptr [r14 + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [r14 + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [r14 + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [r14 + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [r14 + 120], xmm0
	xorps xmm0, xmm0
	movups xmmword ptr [r14 + 136], xmm0
	movups xmmword ptr [r14 + 152], xmm0
	movups xmmword ptr [r14 + 168], xmm0
	movups xmmword ptr [r14 + 184], xmm0
	movups xmmword ptr [r14 + 200], xmm0
	movups xmmword ptr [r14 + 216], xmm0
	movups xmmword ptr [r14 + 232], xmm0
	movups xmmword ptr [r14 + 248], xmm0
	mov qword ptr [r14 + 264], 0
.Ltmp18146:
	xor eax, eax
	mov rcx, qword ptr [rsp + 24]
	mov dl, 1
	lock cmpxchg	byte ptr [rcx], dl
.Ltmp18147:
	jne .LBB170_74
.Ltmp18148:
.LBB170_55:
	mov rax, qword ptr [rsp + 56]
.Ltmp18149:
	mov r13, qword ptr [rax + 1016]
.Ltmp18150:
	cmp r13, qword ptr [rax + 1000]
	jne .LBB170_57
.Ltmp17921:
	mov rdi, qword ptr [rsp + 96]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp17922:
.LBB170_57:
	mov rcx, qword ptr [rsp + 56]
.Ltmp18153:
	mov rax, qword ptr [rcx + 1008]
.Ltmp18154:
	mov qword ptr [rax + 8*r13], r14
.Ltmp18155:
	inc r13
.Ltmp18156:
	mov qword ptr [rcx + 1016], r13
.Ltmp18157:
	mov al, 1
	xor edx, edx
	lock cmpxchg	byte ptr [rcx + 992], dl
.Ltmp18158:
	jne .LBB170_75
.Ltmp18159:
.LBB170_58:
.Ltmp17929:
	mov rdi, r15
	mov rsi, r14
	mov rdx, r14
	mov rcx, rbx
	mov r8, qword ptr [rsp + 80]
	mov r9, qword ptr [rsp + 40]
	call masstree::internode::InternodeNode<S,_>::split_into
	mov qword ptr [rsp + 80], rax
.Ltmp18160:
	lea rax, [r14 + 136]
.Ltmp18161:
	cmp dword ptr [r15 + 8], 0
.Ltmp18162:
	movzx ecx, byte ptr [r14 + 4]
.Ltmp18163:
	movzx ecx, cl
.Ltmp18164:
	je .LBB170_67
	xor edi, edi
.Ltmp18166:
	xor esi, esi
.Ltmp18167:
	.p2align	4
.LBB170_61:
	cmp rdi, rcx
.Ltmp18169:
	adc rsi, 0
.Ltmp18170:
	cmp rdi, 15
	jae .LBB170_63
.Ltmp18171:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp18172:
	test r8, r8
.Ltmp18173:
	jne .LBB170_64
	jmp .LBB170_65
.Ltmp18174:
	.p2align	4
.LBB170_63:
	mov r8, qword ptr [r14 + 256]
.Ltmp18176:
	test r8, r8
.Ltmp18177:
	je .LBB170_65
.Ltmp18178:
.LBB170_64:
	mov qword ptr [r8 + 264], r14
.Ltmp18179:
.LBB170_65:
	cmp rdi, rcx
.Ltmp18180:
	jae .LBB170_20
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB170_61
	jmp .LBB170_20
.Ltmp18182:
	.p2align	4
.LBB170_67:
	xor edi, edi
.Ltmp18184:
	xor esi, esi
.Ltmp18185:
	.p2align	4
.LBB170_68:
	cmp rdi, rcx
.Ltmp18187:
	adc rsi, 0
.Ltmp18188:
	cmp rdi, 15
	jae .LBB170_70
.Ltmp18189:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp18190:
	test r8, r8
.Ltmp18191:
	jne .LBB170_71
	jmp .LBB170_72
.Ltmp18192:
	.p2align	4
.LBB170_70:
	mov r8, qword ptr [r14 + 256]
.Ltmp18194:
	test r8, r8
.Ltmp18195:
	je .LBB170_72
.Ltmp18196:
.LBB170_71:
	mov qword ptr [r8 + 560], r14
.Ltmp18197:
.LBB170_72:
	cmp rdi, rcx
.Ltmp18198:
	jae .LBB170_20
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB170_68
	jmp .LBB170_20
.Ltmp18200:
.LBB170_74:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18201:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB170_55
.Ltmp18202:
.LBB170_75:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18203:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17928:
	jmp .LBB170_58
.Ltmp18204:
.LBB170_76:
	add r12d, 512
.Ltmp18205:
	and r12d, -1342177792
.LBB170_77:
	mov dword ptr [r15], r12d
	mov al, 4
.Ltmp18207:
	jmp .LBB170_109
.Ltmp18208:
.LBB170_78:
	test cl, 1
	je .LBB170_93
.Ltmp18209:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp18210:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp18211:
	test rax, rax
	mov r15, qword ptr [rsp + 56]
.Ltmp18212:
	je .LBB170_126
.Ltmp18213:
	mov rbx, rax
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp18214:
	xorps xmm0, xmm0
.Ltmp18215:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp18216:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp18218:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 40]
.Ltmp18220:
	mov qword ptr [rbx + 144], rax
.Ltmp18221:
	mov byte ptr [rbx + 4], 1
.Ltmp18222:
	lock or	dword ptr [rbx], 1073741824
	mov cl, 1
.Ltmp18223:
	xor eax, eax
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18224:
	jne .LBB170_127
.Ltmp18225:
.LBB170_81:
	mov r14, qword ptr [r15 + 1016]
.Ltmp18226:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB170_83
.Ltmp17969:
	mov rdi, qword ptr [rsp + 96]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp17970:
.Ltmp18228:
.LBB170_83:
	mov rax, qword ptr [r15 + 1008]
.Ltmp18229:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp18230:
	inc r14
.Ltmp18231:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp18232:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp18233:
	jne .LBB170_129
.Ltmp18234:
.LBB170_84:
	#MEMBARRIER
	mov qword ptr [r13 + 560], rbx
	mov rcx, qword ptr [rsp + 40]
.Ltmp18236:
	mov qword ptr [rcx + 560], rbx
.Ltmp18237:
	lock and	dword ptr [r13], -1073741825
	mov r13, rcx
.Ltmp18239:
	jmp .LBB170_106
.Ltmp18240:
.LBB170_85:
	test cl, 1
	je .LBB170_99
.Ltmp18241:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp18242:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp18243:
	test rax, rax
	mov r15, qword ptr [rsp + 56]
.Ltmp18244:
	je .LBB170_126
.Ltmp18245:
	mov rbx, rax
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp18246:
	xorps xmm0, xmm0
.Ltmp18247:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp18248:
	lock or	dword ptr [rax], 1073741824
.Ltmp18249:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp18251:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 40]
.Ltmp18253:
	mov qword ptr [rbx + 144], rax
.Ltmp18254:
	mov byte ptr [rbx + 4], 1
	mov cl, 1
.Ltmp18255:
	xor eax, eax
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18256:
	jne .LBB170_130
.Ltmp18257:
.LBB170_88:
	mov r14, qword ptr [r15 + 1016]
.Ltmp18258:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB170_90
.Ltmp17947:
	mov rdi, qword ptr [rsp + 96]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp17948:
.Ltmp18260:
.LBB170_90:
	mov rax, qword ptr [r15 + 1008]
.Ltmp18261:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp18262:
	inc r14
.Ltmp18263:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp18264:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp18265:
	jne .LBB170_132
.Ltmp18266:
.LBB170_91:
	mov qword ptr [rsp + 112], r13
.Ltmp18267:
	mov rax, r13
	lock cmpxchg	qword ptr [r15 + 1024], rbx
.Ltmp18268:
	jne .LBB170_141
.Ltmp18269:
	#MEMBARRIER
	mov qword ptr [r13 + 560], rbx
	mov rcx, qword ptr [rsp + 40]
.Ltmp18271:
	mov qword ptr [rcx + 560], rbx
.Ltmp18272:
	jmp .LBB170_106
.Ltmp18273:
.LBB170_93:
	mov r14d, dword ptr [r13 + 8]
.Ltmp18274:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp18275:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp18276:
	test rax, rax
	mov r15, qword ptr [rsp + 56]
.Ltmp18277:
	je .LBB170_126
.Ltmp18278:
	mov rbx, rax
	inc r14d
.Ltmp18279:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r14d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp18280:
	xorps xmm0, xmm0
.Ltmp18281:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp18282:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp18284:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 40]
.Ltmp18286:
	mov qword ptr [rbx + 144], rax
.Ltmp18287:
	mov byte ptr [rbx + 4], 1
.Ltmp18288:
	lock or	dword ptr [rbx], 1073741824
	mov cl, 1
.Ltmp18289:
	xor eax, eax
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18290:
	jne .LBB170_128
.Ltmp18291:
.LBB170_95:
	mov r14, qword ptr [r15 + 1016]
.Ltmp18292:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB170_97
.Ltmp17959:
	mov rdi, qword ptr [rsp + 96]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp17960:
.Ltmp18294:
.LBB170_97:
	mov rax, qword ptr [r15 + 1008]
.Ltmp18295:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp18296:
	inc r14
.Ltmp18297:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp18298:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp18299:
	je .LBB170_105
.Ltmp17965:
	mov rdi, qword ptr [rsp + 24]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17966:
	jmp .LBB170_105
.Ltmp18301:
.LBB170_99:
	mov r14d, dword ptr [r13 + 8]
.Ltmp18302:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp18303:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp18304:
	test rax, rax
	mov r15, qword ptr [rsp + 56]
.Ltmp18305:
	je .LBB170_126
.Ltmp18306:
	mov rbx, rax
	inc r14d
.Ltmp18307:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r14d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp18308:
	xorps xmm0, xmm0
.Ltmp18309:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp18310:
	lock or	dword ptr [rax], 1073741824
.Ltmp18311:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp18313:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 40]
.Ltmp18315:
	mov qword ptr [rbx + 144], rax
.Ltmp18316:
	mov byte ptr [rbx + 4], 1
	mov cl, 1
.Ltmp18317:
	xor eax, eax
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18318:
	jne .LBB170_131
.Ltmp18319:
.LBB170_101:
	mov r14, qword ptr [r15 + 1016]
.Ltmp18320:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB170_103
.Ltmp17937:
	mov rdi, qword ptr [rsp + 96]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp17938:
.Ltmp18322:
.LBB170_103:
	mov rax, qword ptr [r15 + 1008]
.Ltmp18323:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp18324:
	inc r14
.Ltmp18325:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp18326:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp18327:
	jne .LBB170_133
.Ltmp18328:
.LBB170_104:
	mov qword ptr [rsp + 112], r13
.Ltmp18329:
	mov rax, r13
	lock cmpxchg	qword ptr [r15 + 1024], rbx
.Ltmp18330:
	jne .LBB170_142
.Ltmp18331:
.LBB170_105:
	#MEMBARRIER
	mov qword ptr [r13 + 264], rbx
	mov rcx, qword ptr [rsp + 40]
	mov qword ptr [rcx + 264], rbx
.Ltmp18333:
.LBB170_106:
	lock and	dword ptr [r13], -1073741825
.Ltmp18334:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
.Ltmp18335:
.LBB170_107:
	mov al, 6
.LBB170_108:
	mov ecx, dword ptr [rsp + 76]
.Ltmp18337:
	add ecx, 512
	and ecx, -1342177792
	mov rdx, qword ptr [rsp + 104]
	mov dword ptr [rdx], ecx
.Ltmp18338:
.LBB170_109:
	lea rsp, [rbp - 40]
.Ltmp18339:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	.cfi_def_cfa rsp, 8
	ret
.Ltmp18340:
.LBB170_110:
	.cfi_def_cfa rbp, 16
	mov rdi, qword ptr [rsp + 40]
.Ltmp18341:
	mov eax, dword ptr [rdi]
	add eax, 512
	and eax, -1342177792
.Ltmp18342:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
.Ltmp18343:
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp18344:
	#MEMBARRIER
	mov dword ptr [rdi], eax
.Ltmp18345:
	and ecx, esi
	mov dword ptr [r15], ecx
	mov al, 4
	jmp .LBB170_108
.Ltmp18347:
.LBB170_111:
	movzx eax, byte ptr [r15 + 4]
.Ltmp18348:
	movzx eax, al
.Ltmp18349:
	cmp rbx, rax
.Ltmp18350:
	jae .LBB170_118
.Ltmp18351:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.206]
	mov rcx, rax
	jmp .LBB170_114
.Ltmp18352:
	.p2align	4
.LBB170_113:
	mov qword ptr [r15 + 8*rsi + 152], rdi
	mov rcx, rsi
.Ltmp18354:
	cmp rbx, rsi
.Ltmp18355:
	jae .LBB170_118
.Ltmp18356:
.LBB170_114:
	cmp rcx, 16
	jae .LBB170_138
.Ltmp18357:
	lea rsi, [rcx - 1]
.Ltmp18358:
	mov rdi, qword ptr [r15 + 8*rsi + 16]
.Ltmp18359:
	cmp rcx, 15
	je .LBB170_139
.Ltmp18360:
	mov qword ptr [r15 + 8*rsi + 24], rdi
.Ltmp18361:
	mov rdi, qword ptr [r15 + 8*rsi + 144]
.Ltmp18362:
	cmp rcx, 14
	jb .LBB170_113
.Ltmp18363:
	mov qword ptr [r15 + 256], rdi
	mov rcx, rsi
.Ltmp18365:
	cmp rbx, rsi
.Ltmp18366:
	jb .LBB170_114
.Ltmp18367:
.LBB170_118:
	cmp rbx, 14
	ja .LBB170_145
	mov rcx, qword ptr [rsp + 80]
.Ltmp18369:
	mov qword ptr [r15 + 8*rbx + 16], rcx
.Ltmp18370:
	jne .LBB170_121
	mov rdi, qword ptr [rsp + 40]
.Ltmp18372:
	mov qword ptr [r15 + 256], rdi
.Ltmp18373:
	jmp .LBB170_122
.LBB170_121:
	mov rdi, qword ptr [rsp + 40]
.Ltmp18375:
	mov qword ptr [r15 + 8*rbx + 144], rdi
.Ltmp18376:
.LBB170_122:
	#MEMBARRIER
	inc al
.Ltmp18377:
	mov byte ptr [r15 + 4], al
	mov rax, qword ptr [rsp + 64]
.Ltmp18379:
	mov qword ptr [rdi + rax], r15
.Ltmp18380:
	mov eax, dword ptr [rdi]
	add eax, 512
	and eax, -1342177792
.Ltmp18381:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp18382:
	#MEMBARRIER
	mov dword ptr [rdi], eax
.Ltmp18383:
	and ecx, esi
	mov dword ptr [r15], ecx
	jmp .LBB170_107
.Ltmp18384:
.LBB170_123:
.Ltmp17932:
	mov edi, 64
	mov esi, 320
	mov r14d, r12d
	call alloc::alloc::handle_alloc_error
.Ltmp17933:
	jmp .LBB170_144
.Ltmp18385:
.LBB170_124:
.Ltmp17980:
	mov edi, 64
	mov esi, 576
	call alloc::alloc::handle_alloc_error
.Ltmp17981:
	jmp .LBB170_144
.Ltmp18386:
.LBB170_125:
.Ltmp17906:
	mov rdi, qword ptr [rsp + 64]
.Ltmp18387:
	call parking_lot::raw_mutex::RawMutex::lock_slow
.Ltmp17907:
	jmp .LBB170_7
.Ltmp18388:
.LBB170_126:
.Ltmp17977:
	mov edi, 64
	mov esi, 320
	call alloc::alloc::handle_alloc_error
	jmp .LBB170_144
.Ltmp18389:
.LBB170_127:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18390:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB170_81
.Ltmp18391:
.LBB170_128:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18392:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB170_95
.Ltmp18393:
.LBB170_129:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18394:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
	jmp .LBB170_84
.Ltmp18395:
.LBB170_130:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18396:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB170_88
.Ltmp18397:
.LBB170_131:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18398:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB170_101
.Ltmp18399:
.LBB170_132:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18400:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
	jmp .LBB170_91
.Ltmp18401:
.LBB170_133:
	mov rdi, qword ptr [rsp + 24]
.Ltmp18402:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17944:
	jmp .LBB170_104
.Ltmp18403:
.LBB170_134:
	mov rcx, qword ptr [rsp + 40]
.Ltmp18404:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
.Ltmp18405:
	mov eax, dword ptr [rsp + 76]
.Ltmp18406:
	add eax, 512
	and eax, -1342177792
	mov rcx, qword ptr [rsp + 104]
	mov dword ptr [rcx], eax
	lea rax, [rsp + 38]
.Ltmp18408:
	mov qword ptr [rsp + 136], rax
	lea rcx, [rip + <bool as core::fmt::Display>::fmt]
	mov qword ptr [rsp + 144], rcx
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.166]
	mov eax, 2
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.164]
.Ltmp18409:
	lea r8, [rsp + 136]
	mov edi, 24
	lea r10, [rsp + 39]
	lea r9, [rsp + 152]
	jmp .LBB170_137
.Ltmp18410:
.LBB170_135:
	mov r15, qword ptr [rsp + 104]
	mov eax, dword ptr [rsp + 76]
	mov r12d, eax
	mov r14, qword ptr [rsp + 40]
.Ltmp18411:
.LBB170_136:
	mov eax, dword ptr [r14]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [r14], eax
.Ltmp18412:
	add r12d, 512
.Ltmp18413:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.171]
	mov eax, 1
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.170]
	lea rcx, [rip + core::fmt::num::imp::<impl core::fmt::Display for u64>::fmt]
	lea r9, [rsp + 136]
	mov edi, 8
	lea r10, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.167]
	mov r8, r9
.Ltmp18414:
.LBB170_137:
	mov qword ptr [r9], r10
	mov qword ptr [r8 + rdi], rcx
	mov rdi, qword ptr [rsp + 176]
	mov qword ptr [rdi], rdx
	mov qword ptr [rdi + 8], 2
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], r8
	mov qword ptr [rdi + 24], rax
	call core::panicking::panic_fmt
.Ltmp18415:
.LBB170_138:
	dec rcx
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.205]
	mov rbx, rcx
.Ltmp18416:
	jmp .LBB170_140
.Ltmp18417:
.LBB170_139:
	mov ebx, 15
.Ltmp18418:
.LBB170_140:
.Ltmp17917:
	mov esi, 15
	mov rdi, rbx
	call core::panicking::panic_bounds_check
.Ltmp18419:
.Ltmp17918:
	jmp .LBB170_144
.Ltmp18420:
.LBB170_141:
	mov qword ptr [rsp + 120], rax
	lea rax, [rsp + 112]
.Ltmp18422:
	mov qword ptr [rsp + 136], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 144], rax
	lea rcx, [rsp + 120]
	mov qword ptr [rsp + 152], rcx
	mov qword ptr [rsp + 160], rax
.Ltmp18423:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.175]
.Ltmp18424:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.177]
	jmp .LBB170_143
.Ltmp18425:
.LBB170_142:
	mov qword ptr [rsp + 120], rax
	lea rax, [rsp + 112]
.Ltmp18427:
	mov qword ptr [rsp + 136], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 144], rax
	lea rcx, [rsp + 120]
	mov qword ptr [rsp + 152], rcx
	mov qword ptr [rsp + 160], rax
.Ltmp18428:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.180]
.Ltmp18429:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.181]
.Ltmp18430:
.LBB170_143:
	lea rax, [rsp + 136]
	lea rdi, [rsp + 192]
	mov qword ptr [rdi + 8], 3
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], rax
	mov qword ptr [rdi + 24], 2
.Ltmp17955:
	call core::panicking::panic_fmt
.Ltmp18431:
.Ltmp17956:
.LBB170_144:
	ud2
.Ltmp18432:
.LBB170_145:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.204]
.Ltmp18433:
	jmp .LBB170_140
.Ltmp18434:
.Ltmp17939:
	mov r13, rax
.Ltmp18435:
	xor ecx, ecx
.Ltmp18436:
	mov al, 1
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18437:
	je .LBB170_176
.Ltmp17940:
	mov rdi, qword ptr [rsp + 24]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17941:
	jmp .LBB170_176
.Ltmp18439:
.Ltmp17942:
	call core::panicking::panic_in_cleanup
.Ltmp18440:
.Ltmp17949:
	mov r13, rax
.Ltmp18441:
	xor ecx, ecx
.Ltmp18442:
	mov al, 1
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18443:
	je .LBB170_176
.Ltmp17950:
	mov rdi, qword ptr [rsp + 24]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17951:
	jmp .LBB170_176
.Ltmp18445:
.Ltmp17952:
	call core::panicking::panic_in_cleanup
.Ltmp18446:
.Ltmp17961:
	mov r13, rax
.Ltmp18447:
	xor ecx, ecx
.Ltmp18448:
	mov al, 1
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18449:
	je .LBB170_176
.Ltmp17962:
	mov rdi, qword ptr [rsp + 24]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17963:
	jmp .LBB170_176
.Ltmp18451:
.Ltmp17964:
	call core::panicking::panic_in_cleanup
.Ltmp18452:
.Ltmp17971:
	mov r13, rax
.Ltmp18453:
	xor ecx, ecx
.Ltmp18454:
	mov al, 1
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18455:
	je .LBB170_176
.Ltmp17972:
	mov rdi, qword ptr [rsp + 24]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17973:
	jmp .LBB170_176
.Ltmp18457:
.Ltmp17974:
	call core::panicking::panic_in_cleanup
.Ltmp18458:
.Ltmp17910:
	mov r13, rax
.Ltmp18459:
	xor ecx, ecx
.Ltmp18460:
	mov al, 1
	mov rdx, qword ptr [rsp + 64]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18461:
	je .LBB170_171
.Ltmp17911:
	mov rdi, qword ptr [rsp + 64]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17912:
	jmp .LBB170_171
.Ltmp18463:
.Ltmp17913:
	call core::panicking::panic_in_cleanup
.Ltmp18464:
.Ltmp17923:
	mov r13, rax
	xor ecx, ecx
.Ltmp18465:
	mov al, 1
	mov rdx, qword ptr [rsp + 24]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp18466:
	je .LBB170_173
.Ltmp17924:
	mov rdi, qword ptr [rsp + 24]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp17925:
	jmp .LBB170_173
.Ltmp18468:
.Ltmp17926:
	call core::panicking::panic_in_cleanup
.Ltmp18469:
.Ltmp17916:
	mov r13, rax
.Ltmp18470:
	jmp .LBB170_171
.Ltmp18471:
.Ltmp17931:
	mov r13, rax
	jmp .LBB170_173
.Ltmp18472:
.Ltmp17903:
	mov r13, rax
.Ltmp18473:
	jmp .LBB170_169
.Ltmp18474:
.Ltmp17979:
	mov r13, rax
	jmp .LBB170_176
.Ltmp18475:
.Ltmp17982:
	mov r13, rax
.Ltmp18476:
	lea rdi, [rsp + 192]
.Ltmp18477:
	call core::ptr::drop_in_place<masstree::leaf24::LeafNode24<masstree::value::LeafValue<u64>>>
.Ltmp18478:
.LBB170_169:
	test r12b, 4
	jne .LBB170_171
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
	mov rdi, r13
	call _Unwind_Resume@PLT
.Ltmp18480:
.LBB170_171:
	add r12d, 512
.Ltmp18481:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	mov rdi, r13
	call _Unwind_Resume@PLT
.Ltmp18482:
.Ltmp17934:
	mov r13, rax
	mov r12d, r14d
.Ltmp18483:
.LBB170_173:
	test r12b, 4
	jne .LBB170_175
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
.Ltmp18485:
	jmp .LBB170_176
.Ltmp18486:
.LBB170_175:
	add r12d, 512
	and r12d, -1342177792
	mov dword ptr [r15], r12d
.Ltmp18487:
.LBB170_176:
	mov eax, dword ptr [rsp + 76]
.Ltmp18488:
	add eax, 512
	and eax, -1342177792
	mov rcx, qword ptr [rsp + 104]
	mov dword ptr [rcx], eax
	mov rdi, r13
	call _Unwind_Resume@PLT
.Ltmp18489:
.Lfunc_end170:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic, .Lfunc_end170-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic","a",@progbits
	.p2align	2, 0x0
GCC_except_table170:
masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic:
.Lfunc_begin214:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception87
	push rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov rbp, rsp
	.cfi_def_cfa_register rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	and rsp, -64
	sub rsp, 704
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	mov rbx, r9
	mov r12d, ecx
	mov r15, rdx
	mov r13, rsi
	mov qword ptr [rsp + 96], rdi
.Ltmp22976:
.Ltmp22952:
	lea rdi, [rsp + 192]
.Ltmp22977:
	mov rdx, r8
.Ltmp22978:
	call <masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point
.Ltmp22979:
.Ltmp22953:
	cmp dword ptr [rsp + 192], 1
	jne .LBB214_13
	mov rax, qword ptr [rsp + 200]
	mov qword ptr [rsp + 112], rax
.Ltmp22982:
	mov ecx, dword ptr [r13]
	mov dword ptr [rsp + 68], 0
.Ltmp22983:
	mov rax, qword ptr [r13 + 344]
.Ltmp22984:
	mov dword ptr [rsp + 92], 0
.Ltmp22985:
	test ecx, 1073741824
.Ltmp22986:
	je .LBB214_4
.Ltmp22987:
	test rax, rax
	sete al
.Ltmp22988:
	mov rcx, qword ptr [rsp + 96]
.Ltmp22989:
	mov rcx, qword ptr [rcx + 992]
.Ltmp22990:
	cmp rcx, r13
	sete cl
.Ltmp22991:
	mov dword ptr [rsp + 92], ecx
.Ltmp22992:
	setne cl
.Ltmp22993:
	and cl, al
	mov dword ptr [rsp + 68], ecx
.Ltmp22994:
.LBB214_4:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 213], xmm0
	movups xmmword ptr [rsp + 197], xmm0
.Ltmp22995:
	movups xmmword ptr [rsp + 392], xmm0
	movups xmmword ptr [rsp + 408], xmm0
	movups xmmword ptr [rsp + 424], xmm0
	movups xmmword ptr [rsp + 440], xmm0
	movups xmmword ptr [rsp + 456], xmm0
	movups xmmword ptr [rsp + 472], xmm0
	movups xmmword ptr [rsp + 488], xmm0
	mov qword ptr [rsp + 504], 0
.Ltmp22996:
	movaps xmmword ptr [rsp + 256], xmm0
	movaps xmmword ptr [rsp + 272], xmm0
	movaps xmmword ptr [rsp + 288], xmm0
	movaps xmmword ptr [rsp + 304], xmm0
	movaps xmmword ptr [rsp + 320], xmm0
	movaps xmmword ptr [rsp + 336], xmm0
	movaps xmmword ptr [rsp + 352], xmm0
	movaps xmmword ptr [rsp + 368], xmm0
	mov qword ptr [rsp + 383], 0
.Ltmp22997:
	mov dword ptr [rsp + 192], -2147483648
	mov byte ptr [rsp + 196], 0
	movabs rax, 81985529216486880
	mov qword ptr [rsp + 248], rax
	movaps xmmword ptr [rsp + 512], xmm0
	movaps xmmword ptr [rsp + 528], xmm0
.Ltmp22998:
	mov edi, 384
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp22999:
	test rax, rax
	je .LBB214_117
.Ltmp23000:
	mov r14, rax
	lea rsi, [rsp + 192]
	mov edx, 384
	mov rdi, rax
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp23001:
	or dword ptr [r15], 4
.Ltmp23002:
	#MEMBARRIER
.Ltmp22955:
	lea rdi, [rsp + 192]
	mov qword ptr [rsp + 624], rdi
.Ltmp23003:
	mov rsi, r13
	mov rdx, qword ptr [rsp + 112]
	mov rcx, r14
	mov r8, rbx
	call <masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated
.Ltmp22956:
	mov r14, qword ptr [rsp + 192]
.Ltmp23005:
	mov rax, qword ptr [rsp + 200]
.Ltmp23006:
	mov qword ptr [rsp + 104], rax
.Ltmp23007:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23008:
	test rax, rax
	mov ebx, dword ptr [rsp + 92]
.Ltmp23009:
	je .LBB214_118
.Ltmp23010:
	mov rcx, rax
	mov qword ptr [rax], r14
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 96]
.Ltmp23012:
	mov rax, qword ptr [rdx + 960]
.Ltmp23013:
	mov qword ptr [rcx + 8], rax
.Ltmp23014:
	lock cmpxchg	qword ptr [rdx + 960], rcx
.Ltmp23015:
	je .LBB214_9
.Ltmp23016:
	.p2align	4
.LBB214_8:
	pause
.Ltmp23018:
	mov rax, qword ptr [rdx + 960]
.Ltmp23019:
	mov qword ptr [rcx + 8], rax
.Ltmp23020:
	lock cmpxchg	qword ptr [rdx + 960], rcx
.Ltmp23021:
	jne .LBB214_8
.Ltmp23022:
.LBB214_9:
	lock inc	qword ptr [rdx + 968]
.Ltmp23023:
	mov rcx, qword ptr [r13 + 328]
.Ltmp23024:
	test cl, 1
	je .LBB214_11
	jmp .LBB214_10
.Ltmp23025:
	.p2align	4
.LBB214_12:
	pause
.Ltmp23027:
	mov rcx, qword ptr [r13 + 328]
.Ltmp23028:
	test cl, 1
	je .LBB214_11
.Ltmp23029:
.LBB214_10:
	mov rdi, r13
	call masstree::leaf15::LeafNode15<S>::wait_for_split
.Ltmp23030:
	mov rcx, qword ptr [r13 + 328]
.Ltmp23031:
	test cl, 1
	jne .LBB214_10
.Ltmp23032:
.LBB214_11:
	lea rdx, [rcx + 1]
.Ltmp23033:
	mov rax, rcx
	lock cmpxchg	qword ptr [r13 + 328], rdx
.Ltmp23034:
	jne .LBB214_12
.Ltmp23035:
	mov qword ptr [r14 + 336], r13
.Ltmp23036:
	mov qword ptr [r14 + 328], rcx
.Ltmp23037:
	test rcx, rcx
.Ltmp23038:
	je .LBB214_17
.Ltmp23039:
	mov qword ptr [rcx + 336], r14
.Ltmp23040:
.LBB214_17:
	#MEMBARRIER
	mov qword ptr [r13 + 328], r14
	mov byte ptr [rsp + 62], bl
	mov eax, dword ptr [rsp + 68]
	mov byte ptr [rsp + 63], al
	mov cl, 1
	mov eax, 1
	mov esi, 344
	xorps xmm0, xmm0
	mov r9, qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	mov r10d, -268435464
	jmp .LBB214_19
.Ltmp23042:
	.p2align	4
.LBB214_18:
	test dl, 1
	mov rax, r14
	cmovne rax, r15
.Ltmp23043:
	mov qword ptr [r9 + r11], rax
.Ltmp23044:
	mov eax, dword ptr [r9]
.Ltmp23045:
	add eax, 512
	and eax, -1342177792
.Ltmp23046:
	mov ecx, dword ptr [rsp + 64]
.Ltmp23047:
	add ecx, 512
	and ecx, -1342177792
.Ltmp23048:
	cmp qword ptr [rsp + 632], r15
.Ltmp23049:
	#MEMBARRIER
	mov dword ptr [r9], eax
	mov rax, qword ptr [rsp + 80]
.Ltmp23050:
	mov dword ptr [rax], ecx
.Ltmp23051:
	sete byte ptr [rsp + 62]
	mov eax, dword ptr [rsp + 68]
	mov byte ptr [rsp + 63], al
.Ltmp23052:
	mov rdx, qword ptr [rsp + 112]
.Ltmp23053:
	lea rax, [rdx + 1]
	xor ecx, ecx
	mov r13, r15
	cmp rdx, 63
	mov esi, 344
	mov r9, qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	ja .LBB214_121
.Ltmp23054:
.LBB214_19:
	mov r8, r14
.Ltmp23055:
	mov dword ptr [rsp + 64], r12d
.Ltmp23056:
	test cl, 1
	mov r11d, 264
	cmovne r11, rsi
	mov qword ptr [rsp + 80], r15
.Ltmp23057:
	not bl
	mov rsi, rax
	xor edx, edx
	mov qword ptr [rsp + 72], r14
.Ltmp23058:
	jmp .LBB214_23
.Ltmp23059:
	.p2align	4
.LBB214_20:
	mov rax, qword ptr [r13 + r11]
.Ltmp23060:
	cmp rax, r15
	je .LBB214_34
	xor edx, edx
.Ltmp23062:
.LBB214_22:
	lea eax, [r12 + 512]
	mov esi, r12d
	add esi, 8
	test r12b, 4
	mov edi, -1342177792
	cmove edi, r10d
	cmovne esi, eax
	and esi, edi
	mov dword ptr [r15], esi
.Ltmp23063:
	pause
.Ltmp23064:
	mov rax, qword ptr [rsp + 112]
.Ltmp23065:
	cmp rax, 63
	lea rsi, [rax + 1]
.Ltmp23066:
	ja .LBB214_120
.Ltmp23067:
.LBB214_23:
	mov r15, qword ptr [r13 + r11]
.Ltmp23068:
	test r15, r15
.Ltmp23069:
	jne .LBB214_25
.Ltmp23070:
	test byte ptr [rsp + 68], 1
	jne .LBB214_73
.Ltmp23071:
.LBB214_25:
	test r15, r15
	setne al
.Ltmp23072:
	or al, bl
	test al, 1
	je .LBB214_79
.Ltmp23073:
	test r15, r15
.Ltmp23074:
	je .LBB214_119
.Ltmp23075:
	mov qword ptr [rsp + 112], rsi
	xor esi, esi
	jmp .LBB214_29
.Ltmp23076:
	.p2align	4
.LBB214_28:
	and esi, 7
	lea esi, [2*rsi + 1]
.LBB214_29:
	mov r12d, dword ptr [r15]
	test r12b, 1
	jne .LBB214_31
	mov r14d, r12d
	or r14d, 3
	mov eax, r12d
	lock cmpxchg	dword ptr [r15], r14d
	je .LBB214_20
.LBB214_31:
	xor eax, eax
	.p2align	4
.LBB214_32:
	mov edi, eax
	pause
	cmp eax, esi
	adc eax, 0
	cmp edi, esi
	jae .LBB214_28
	cmp eax, esi
	jbe .LBB214_32
	jmp .LBB214_28
.Ltmp23082:
	.p2align	4
.LBB214_34:
	mov edi, ebx
.Ltmp23084:
	movzx eax, byte ptr [r15 + 4]
.Ltmp23085:
	movzx ebx, al
.Ltmp23086:
	test bl, bl
.Ltmp23087:
	je .LBB214_41
	xor eax, eax
	jmp .LBB214_36
.Ltmp23089:
	.p2align	4
.LBB214_38:
	mov rsi, qword ptr [r15 + 256]
.Ltmp23090:
	cmp rsi, r13
.Ltmp23091:
	je .LBB214_44
.Ltmp23092:
.LBB214_39:
	inc rax
.Ltmp23093:
	cmp rbx, rax
.Ltmp23094:
	je .LBB214_40
.Ltmp23095:
.LBB214_36:
	cmp rax, 15
	jae .LBB214_38
.Ltmp23096:
	mov rsi, qword ptr [r15 + 8*rax + 136]
.Ltmp23097:
	cmp rsi, r13
.Ltmp23098:
	jne .LBB214_39
	jmp .LBB214_44
.Ltmp23099:
.LBB214_40:
	dec rax
.Ltmp23100:
	cmp rax, 14
	jae .LBB214_42
.Ltmp23101:
.LBB214_41:
	mov rax, qword ptr [r15 + 8*rbx + 136]
.Ltmp23102:
	cmp rax, r13
.Ltmp23103:
	jne .LBB214_43
	jmp .LBB214_45
.Ltmp23104:
.LBB214_42:
	mov rax, qword ptr [r15 + 256]
.Ltmp23105:
	cmp rax, r13
.Ltmp23106:
	je .LBB214_45
.Ltmp23107:
.LBB214_43:
	inc rdx
	cmp rdx, 16
	mov ebx, edi
	jbe .LBB214_22
	jmp .LBB214_102
	.p2align	4
.LBB214_44:
	mov rbx, rax
.Ltmp23111:
.LBB214_45:
	movzx eax, byte ptr [r15 + 4]
.Ltmp23112:
	cmp al, 14
	mov rax, qword ptr [rsp + 96]
.Ltmp23113:
	jbe .LBB214_103
.Ltmp23114:
	or dword ptr [r15], 4
.Ltmp23115:
	#MEMBARRIER
	mov rcx, qword ptr [rax + 992]
.Ltmp23116:
	cmp rcx, r15
	sete al
	mov dword ptr [rsp + 92], eax
.Ltmp23117:
	mov qword ptr [rsp + 640], r11
	mov qword ptr [rsp + 632], rcx
.Ltmp23118:
	je .LBB214_49
.Ltmp23119:
	mov rax, qword ptr [r15 + 264]
.Ltmp23120:
	test rax, rax
.Ltmp23121:
	je .LBB214_50
.LBB214_49:
	mov dword ptr [rsp + 68], 0
	jmp .LBB214_51
.Ltmp23123:
.LBB214_50:
	mov eax, dword ptr [r15]
.Ltmp23124:
	shr eax, 30
.Ltmp23125:
	and al, 1
	mov dword ptr [rsp + 68], eax
.Ltmp23126:
.LBB214_51:
	or r12d, 7
.Ltmp23127:
	mov eax, dword ptr [r15 + 8]
	mov dword ptr [rsp + 140], eax
.Ltmp23129:
	mov r13d, dword ptr [r15]
.Ltmp23130:
	lea rax, [rsp + 196]
.Ltmp23131:
	movups xmmword ptr [rax + 96], xmm0
	movups xmmword ptr [rax + 80], xmm0
	movups xmmword ptr [rax + 64], xmm0
	movups xmmword ptr [rax + 48], xmm0
	movups xmmword ptr [rax + 32], xmm0
	movups xmmword ptr [rax + 16], xmm0
	movups xmmword ptr [rax], xmm0
	mov qword ptr [rax + 112], 0
.Ltmp23132:
	mov eax, 320
	mov qword ptr [rsp + 184], rax
	mov eax, 64
	mov qword ptr [rsp + 176], rax
.Ltmp23133:
	mov edi, 320
	mov esi, 64
	call r9
.Ltmp23134:
	test rax, rax
	je .LBB214_115
.Ltmp23135:
	mov r14, rax
	and r13d, -2147483648
	or r13d, 5
.Ltmp23136:
	mov dword ptr [rax], r13d
	mov byte ptr [rax + 4], 0
	mov eax, dword ptr [rsp + 140]
	mov dword ptr [r14 + 8], eax
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [r14 + 12], xmm0
	movups xmmword ptr [r14 + 28], xmm1
	movups xmmword ptr [r14 + 44], xmm2
	movups xmmword ptr [r14 + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [r14 + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [r14 + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [r14 + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [r14 + 120], xmm0
	xorps xmm0, xmm0
	movups xmmword ptr [r14 + 136], xmm0
	movups xmmword ptr [r14 + 152], xmm0
	movups xmmword ptr [r14 + 168], xmm0
	movups xmmword ptr [r14 + 184], xmm0
	movups xmmword ptr [r14 + 200], xmm0
	movups xmmword ptr [r14 + 216], xmm0
	movups xmmword ptr [r14 + 232], xmm0
	movups xmmword ptr [r14 + 248], xmm0
	mov qword ptr [r14 + 264], 0
.Ltmp23137:
	mov eax, 16
	mov qword ptr [rsp + 184], rax
	mov eax, 8
	mov qword ptr [rsp + 176], rax
.Ltmp23138:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23139:
	test rax, rax
	je .LBB214_115
.Ltmp23140:
	mov rcx, rax
	mov qword ptr [rax], r14
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 96]
.Ltmp23142:
	mov rax, qword ptr [rdx + 976]
.Ltmp23143:
	mov qword ptr [rcx + 8], rax
.Ltmp23144:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23145:
	je .LBB214_55
.Ltmp23146:
	.p2align	4
.LBB214_54:
	pause
.Ltmp23148:
	mov rax, qword ptr [rdx + 976]
.Ltmp23149:
	mov qword ptr [rcx + 8], rax
.Ltmp23150:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23151:
	jne .LBB214_54
.Ltmp23152:
.LBB214_55:
	lock inc	qword ptr [rdx + 984]
.Ltmp23153:
.Ltmp22959:
	mov rdi, r15
	mov rsi, r14
	mov rdx, r14
	mov rcx, rbx
	mov r8, qword ptr [rsp + 104]
	mov r9, qword ptr [rsp + 72]
	call masstree::internode::InternodeNode<S,_>::split_into
.Ltmp23154:
.Ltmp22960:
	mov qword ptr [rsp + 104], rax
.Ltmp23155:
	lea rax, [r14 + 136]
.Ltmp23156:
	cmp dword ptr [r15 + 8], 0
.Ltmp23157:
	movzx ecx, byte ptr [r14 + 4]
.Ltmp23158:
	movzx ecx, cl
	mov ebx, dword ptr [rsp + 92]
.Ltmp23159:
	xorps xmm0, xmm0
	mov r10d, -268435464
	mov r11, qword ptr [rsp + 640]
.Ltmp23160:
	je .LBB214_64
	xor edi, edi
.Ltmp23162:
	xor esi, esi
	mov r9, qword ptr [rsp + 72]
.Ltmp23163:
	.p2align	4
.LBB214_58:
	cmp rdi, rcx
.Ltmp23165:
	adc rsi, 0
.Ltmp23166:
	cmp rdi, 15
	jae .LBB214_60
.Ltmp23167:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp23168:
	test r8, r8
.Ltmp23169:
	jne .LBB214_61
	jmp .LBB214_62
.Ltmp23170:
	.p2align	4
.LBB214_60:
	mov r8, qword ptr [r14 + 256]
.Ltmp23172:
	test r8, r8
.Ltmp23173:
	je .LBB214_62
.Ltmp23174:
.LBB214_61:
	mov qword ptr [r8 + 264], r14
.Ltmp23175:
.LBB214_62:
	cmp rdi, rcx
.Ltmp23176:
	jae .LBB214_18
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB214_58
	jmp .LBB214_18
.Ltmp23178:
	.p2align	4
.LBB214_64:
	xor edi, edi
.Ltmp23180:
	xor esi, esi
	mov r9, qword ptr [rsp + 72]
.Ltmp23181:
	.p2align	4
.LBB214_65:
	cmp rdi, rcx
.Ltmp23183:
	adc rsi, 0
.Ltmp23184:
	cmp rdi, 15
	jae .LBB214_67
.Ltmp23185:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp23186:
	test r8, r8
.Ltmp23187:
	jne .LBB214_68
	jmp .LBB214_69
.Ltmp23188:
	.p2align	4
.LBB214_67:
	mov r8, qword ptr [r14 + 256]
.Ltmp23190:
	test r8, r8
.Ltmp23191:
	je .LBB214_69
.Ltmp23192:
.LBB214_68:
	mov qword ptr [r8 + 344], r14
.Ltmp23193:
.LBB214_69:
	cmp rdi, rcx
.Ltmp23194:
	jae .LBB214_18
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB214_65
	jmp .LBB214_18
.Ltmp23196:
.LBB214_13:
	test r12b, 4
	jne .LBB214_71
	mov eax, r12d
	and eax, 2
	lea r12d, [r12 + 4*rax]
.Ltmp23198:
	and r12d, -268435464
	jmp .LBB214_72
.Ltmp23199:
.LBB214_71:
	add r12d, 512
.Ltmp23200:
	and r12d, -1342177792
.LBB214_72:
	mov dword ptr [r15], r12d
	mov al, 4
.Ltmp23202:
	jmp .LBB214_101
.Ltmp23203:
.LBB214_73:
	test cl, 1
	je .LBB214_86
.Ltmp23204:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp23205:
	mov ebx, 320
	mov r14d, 64
.Ltmp23206:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23207:
	test rax, rax
	je .LBB214_116
.Ltmp23208:
	mov r15, rax
.Ltmp23209:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp23210:
	xorps xmm0, xmm0
.Ltmp23211:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp23212:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 104]
.Ltmp23214:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 72]
.Ltmp23216:
	mov qword ptr [r15 + 144], rax
.Ltmp23217:
	mov byte ptr [r15 + 4], 1
.Ltmp23218:
	lock or	dword ptr [r15], 1073741824
.Ltmp23219:
	mov ebx, 16
	mov r14d, 8
.Ltmp23220:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23221:
	test rax, rax
	je .LBB214_116
.Ltmp23222:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 96]
.Ltmp23224:
	mov rax, qword ptr [rdx + 976]
.Ltmp23225:
	mov qword ptr [rcx + 8], rax
.Ltmp23226:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23227:
	je .LBB214_78
.Ltmp23228:
	.p2align	4
.LBB214_77:
	pause
.Ltmp23230:
	mov rax, qword ptr [rdx + 976]
.Ltmp23231:
	mov qword ptr [rcx + 8], rax
.Ltmp23232:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23233:
	jne .LBB214_77
.Ltmp23234:
.LBB214_78:
	lock inc	qword ptr [rdx + 984]
.Ltmp23235:
	#MEMBARRIER
	mov qword ptr [r13 + 344], r15
	mov rcx, qword ptr [rsp + 72]
.Ltmp23237:
	mov qword ptr [rcx + 344], r15
.Ltmp23238:
	lock and	dword ptr [r13], -1073741825
	mov r13, rcx
.Ltmp23240:
	jmp .LBB214_98
.Ltmp23241:
.LBB214_79:
	test cl, 1
	je .LBB214_91
.Ltmp23242:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp23243:
	mov ebx, 320
	mov r14d, 64
.Ltmp23244:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23245:
	test rax, rax
	je .LBB214_116
.Ltmp23246:
	mov r15, rax
.Ltmp23247:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp23248:
	xorps xmm0, xmm0
.Ltmp23249:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp23250:
	lock or	dword ptr [rax], 1073741824
.Ltmp23251:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 104]
.Ltmp23253:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 72]
.Ltmp23255:
	mov qword ptr [r15 + 144], rax
.Ltmp23256:
	mov byte ptr [r15 + 4], 1
.Ltmp23257:
	mov ebx, 16
	mov r14d, 8
.Ltmp23258:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23259:
	test rax, rax
	je .LBB214_116
.Ltmp23260:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 96]
.Ltmp23262:
	mov rax, qword ptr [rdx + 976]
.Ltmp23263:
	mov qword ptr [rcx + 8], rax
.Ltmp23264:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23265:
	je .LBB214_84
.Ltmp23266:
	.p2align	4
.LBB214_83:
	pause
.Ltmp23268:
	mov rax, qword ptr [rdx + 976]
.Ltmp23269:
	mov qword ptr [rcx + 8], rax
.Ltmp23270:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23271:
	jne .LBB214_83
.Ltmp23272:
.LBB214_84:
	lock inc	qword ptr [rdx + 984]
.Ltmp23273:
	mov qword ptr [rsp + 120], r13
.Ltmp23274:
	mov rax, r13
	lock cmpxchg	qword ptr [rdx + 992], r15
	mov rcx, qword ptr [rsp + 72]
.Ltmp23275:
	jne .LBB214_126
.Ltmp23276:
	#MEMBARRIER
	mov qword ptr [r13 + 344], r15
.Ltmp23277:
	mov qword ptr [rcx + 344], r15
.Ltmp23278:
	jmp .LBB214_98
.Ltmp23279:
.LBB214_86:
	mov r12d, dword ptr [r13 + 8]
.Ltmp23280:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp23281:
	mov ebx, 320
	mov r14d, 64
.Ltmp23282:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23283:
	test rax, rax
	je .LBB214_116
.Ltmp23284:
	mov r15, rax
.Ltmp23285:
	inc r12d
.Ltmp23286:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r12d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp23287:
	xorps xmm0, xmm0
.Ltmp23288:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp23289:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 104]
.Ltmp23291:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 72]
.Ltmp23293:
	mov qword ptr [r15 + 144], rax
.Ltmp23294:
	mov byte ptr [r15 + 4], 1
.Ltmp23295:
	lock or	dword ptr [r15], 1073741824
.Ltmp23296:
	mov ebx, 16
	mov r14d, 8
.Ltmp23297:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23298:
	test rax, rax
	je .LBB214_116
.Ltmp23299:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 96]
.Ltmp23301:
	mov rax, qword ptr [rdx + 976]
.Ltmp23302:
	mov qword ptr [rcx + 8], rax
.Ltmp23303:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23304:
	je .LBB214_90
.Ltmp23305:
	.p2align	4
.LBB214_89:
	pause
.Ltmp23307:
	mov rax, qword ptr [rdx + 976]
.Ltmp23308:
	mov qword ptr [rcx + 8], rax
.Ltmp23309:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23310:
	jne .LBB214_89
.Ltmp23311:
.LBB214_90:
	lock inc	qword ptr [rdx + 984]
.Ltmp23312:
	#MEMBARRIER
	mov qword ptr [r13 + 264], r15
	mov rcx, qword ptr [rsp + 72]
.Ltmp23314:
	jmp .LBB214_97
.Ltmp23315:
.LBB214_91:
	mov r12d, dword ptr [r13 + 8]
.Ltmp23316:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp23317:
	mov ebx, 320
	mov r14d, 64
.Ltmp23318:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23319:
	test rax, rax
	je .LBB214_116
.Ltmp23320:
	mov r15, rax
.Ltmp23321:
	inc r12d
.Ltmp23322:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r12d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp23323:
	xorps xmm0, xmm0
.Ltmp23324:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp23325:
	lock or	dword ptr [rax], 1073741824
.Ltmp23326:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 104]
.Ltmp23328:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 72]
.Ltmp23330:
	mov qword ptr [r15 + 144], rax
.Ltmp23331:
	mov byte ptr [r15 + 4], 1
.Ltmp23332:
	mov ebx, 16
	mov r14d, 8
.Ltmp23333:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp23334:
	test rax, rax
	je .LBB214_116
.Ltmp23335:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 96]
.Ltmp23337:
	mov rax, qword ptr [rdx + 976]
.Ltmp23338:
	mov qword ptr [rcx + 8], rax
.Ltmp23339:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23340:
	je .LBB214_95
.Ltmp23341:
	.p2align	4
.LBB214_94:
	pause
.Ltmp23343:
	mov rax, qword ptr [rdx + 976]
.Ltmp23344:
	mov qword ptr [rcx + 8], rax
.Ltmp23345:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp23346:
	jne .LBB214_94
.Ltmp23347:
.LBB214_95:
	lock inc	qword ptr [rdx + 984]
.Ltmp23348:
	mov qword ptr [rsp + 120], r13
.Ltmp23349:
	mov rax, r13
	lock cmpxchg	qword ptr [rdx + 992], r15
	mov rcx, qword ptr [rsp + 72]
.Ltmp23350:
	jne .LBB214_127
.Ltmp23351:
	#MEMBARRIER
	mov qword ptr [r13 + 264], r15
.Ltmp23352:
.LBB214_97:
	mov qword ptr [rcx + 264], r15
.Ltmp23353:
.LBB214_98:
	lock and	dword ptr [r13], -1073741825
.Ltmp23354:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
.Ltmp23355:
.LBB214_99:
	mov al, 6
.LBB214_100:
	mov ecx, dword ptr [rsp + 64]
	mov rdx, qword ptr [rsp + 80]
.Ltmp23357:
	add ecx, 512
	and ecx, -1342177792
	mov dword ptr [rdx], ecx
.Ltmp23358:
.LBB214_101:
	lea rsp, [rbp - 40]
.Ltmp23359:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	.cfi_def_cfa rsp, 8
	ret
.Ltmp23360:
.LBB214_102:
	.cfi_def_cfa rbp, 16
	mov eax, dword ptr [r8]
	add eax, 512
	and eax, -1342177792
.Ltmp23361:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
.Ltmp23362:
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp23363:
	#MEMBARRIER
	mov dword ptr [r8], eax
.Ltmp23364:
	and ecx, esi
	mov dword ptr [r15], ecx
	mov al, 4
	jmp .LBB214_100
.Ltmp23366:
.LBB214_103:
	movzx eax, byte ptr [r15 + 4]
.Ltmp23367:
	movzx eax, al
.Ltmp23368:
	cmp rbx, rax
.Ltmp23369:
	jae .LBB214_110
.Ltmp23370:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.206]
	mov rcx, rax
	jmp .LBB214_106
.Ltmp23371:
	.p2align	4
.LBB214_105:
	mov qword ptr [r15 + 8*rsi + 152], rdi
	mov rcx, rsi
.Ltmp23373:
	cmp rbx, rsi
.Ltmp23374:
	jae .LBB214_110
.Ltmp23375:
.LBB214_106:
	cmp rcx, 16
	jae .LBB214_123
.Ltmp23376:
	lea rsi, [rcx - 1]
.Ltmp23377:
	mov rdi, qword ptr [r15 + 8*rsi + 16]
.Ltmp23378:
	cmp rcx, 15
	je .LBB214_124
.Ltmp23379:
	mov qword ptr [r15 + 8*rsi + 24], rdi
.Ltmp23380:
	mov rdi, qword ptr [r15 + 8*rsi + 144]
.Ltmp23381:
	cmp rcx, 14
	jb .LBB214_105
.Ltmp23382:
	mov qword ptr [r15 + 256], rdi
	mov rcx, rsi
.Ltmp23384:
	cmp rbx, rsi
.Ltmp23385:
	jb .LBB214_106
.Ltmp23386:
.LBB214_110:
	cmp rbx, 14
	ja .LBB214_130
	mov rcx, qword ptr [rsp + 104]
.Ltmp23388:
	mov qword ptr [r15 + 8*rbx + 16], rcx
.Ltmp23389:
	jne .LBB214_113
.Ltmp23390:
	mov qword ptr [r15 + 256], r8
.Ltmp23391:
	jmp .LBB214_114
.Ltmp23392:
.LBB214_113:
	mov qword ptr [r15 + 8*rbx + 144], r8
.Ltmp23393:
.LBB214_114:
	#MEMBARRIER
	inc al
.Ltmp23394:
	mov byte ptr [r15 + 4], al
.Ltmp23395:
	mov qword ptr [r8 + r11], r15
.Ltmp23396:
	mov eax, dword ptr [r8]
	add eax, 512
	and eax, -1342177792
.Ltmp23398:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp23399:
	#MEMBARRIER
	mov dword ptr [r8], eax
.Ltmp23400:
	and ecx, esi
	mov dword ptr [r15], ecx
	jmp .LBB214_99
.Ltmp23401:
.LBB214_115:
.Ltmp22962:
	mov r14d, r12d
	mov rdi, qword ptr [rsp + 176]
	mov rsi, qword ptr [rsp + 184]
.Ltmp23402:
	call alloc::alloc::handle_alloc_error
.Ltmp22963:
	jmp .LBB214_129
.Ltmp23403:
.LBB214_116:
.Ltmp22967:
	mov rdi, r14
	mov rsi, rbx
	call alloc::alloc::handle_alloc_error
.Ltmp22968:
	jmp .LBB214_129
.Ltmp23404:
.LBB214_117:
.Ltmp22973:
	mov edi, 64
	mov esi, 384
	call alloc::alloc::handle_alloc_error
.Ltmp22974:
	jmp .LBB214_129
.Ltmp23405:
.LBB214_118:
.Ltmp22970:
	mov edi, 8
	mov esi, 16
	call alloc::alloc::handle_alloc_error
.Ltmp22971:
	jmp .LBB214_129
.Ltmp23406:
.LBB214_119:
	mov eax, dword ptr [r8]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [r8], eax
.Ltmp23407:
	mov eax, dword ptr [rsp + 64]
.Ltmp23408:
	add eax, 512
	and eax, -1342177792
	mov rcx, qword ptr [rsp + 80]
	mov dword ptr [rcx], eax
	lea rax, [rsp + 62]
.Ltmp23410:
	mov qword ptr [rsp + 144], rax
	lea rcx, [rip + <bool as core::fmt::Display>::fmt]
	mov qword ptr [rsp + 152], rcx
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.166]
	mov eax, 2
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.164]
.Ltmp23411:
	lea r8, [rsp + 144]
	mov edi, 24
	lea r10, [rsp + 63]
	lea r9, [rsp + 160]
	jmp .LBB214_122
.Ltmp23412:
.LBB214_120:
	mov r15, qword ptr [rsp + 80]
	mov eax, dword ptr [rsp + 64]
	mov r12d, eax
	mov r14, r8
.Ltmp23413:
.LBB214_121:
	mov eax, dword ptr [r14]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [r14], eax
.Ltmp23414:
	add r12d, 512
.Ltmp23415:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.171]
	mov eax, 1
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.170]
	lea rcx, [rip + core::fmt::num::imp::<impl core::fmt::Display for u64>::fmt]
	lea r9, [rsp + 144]
	mov edi, 8
	lea r10, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.167]
	mov r8, r9
.Ltmp23416:
.LBB214_122:
	mov qword ptr [r9], r10
	mov qword ptr [r8 + rdi], rcx
	mov rdi, qword ptr [rsp + 624]
	mov qword ptr [rdi], rdx
	mov qword ptr [rdi + 8], 2
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], r8
	mov qword ptr [rdi + 24], rax
	call core::panicking::panic_fmt
.Ltmp23417:
.LBB214_123:
	dec rcx
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.205]
	mov rbx, rcx
.Ltmp23418:
	jmp .LBB214_125
.Ltmp23419:
.LBB214_124:
	mov ebx, 15
.Ltmp23420:
.LBB214_125:
.Ltmp22957:
	mov esi, 15
	mov rdi, rbx
	call core::panicking::panic_bounds_check
.Ltmp23421:
.Ltmp22958:
	jmp .LBB214_129
.Ltmp23422:
.LBB214_126:
	mov qword ptr [rsp + 128], rax
	lea rax, [rsp + 120]
.Ltmp23424:
	mov qword ptr [rsp + 144], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 152], rax
	lea rcx, [rsp + 128]
	mov qword ptr [rsp + 160], rcx
	mov qword ptr [rsp + 168], rax
.Ltmp23425:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.175]
.Ltmp23426:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.177]
	jmp .LBB214_128
.Ltmp23427:
.LBB214_127:
	mov qword ptr [rsp + 128], rax
	lea rax, [rsp + 120]
.Ltmp23429:
	mov qword ptr [rsp + 144], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 152], rax
	lea rcx, [rsp + 128]
	mov qword ptr [rsp + 160], rcx
	mov qword ptr [rsp + 168], rax
.Ltmp23430:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.180]
.Ltmp23431:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.181]
.Ltmp23432:
.LBB214_128:
	lea rax, [rsp + 144]
	lea rdi, [rsp + 192]
	mov qword ptr [rdi + 8], 3
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], rax
	mov qword ptr [rdi + 24], 2
.Ltmp22965:
	call core::panicking::panic_fmt
.Ltmp23433:
.Ltmp22966:
.LBB214_129:
	ud2
.Ltmp23434:
.LBB214_130:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.204]
.Ltmp23435:
	jmp .LBB214_125
.Ltmp23436:
.Ltmp22961:
	mov rdi, rax
	jmp .LBB214_140
.Ltmp23437:
.Ltmp22954:
	mov rbx, rax
.Ltmp23438:
	jmp .LBB214_134
.Ltmp23439:
.Ltmp22975:
	mov rbx, rax
.Ltmp23440:
	lea rdi, [rsp + 192]
.Ltmp23441:
	call core::ptr::drop_in_place<masstree::leaf15::LeafNode15<masstree::value::LeafValue<u64>>>
.Ltmp23442:
.LBB214_134:
	test r12b, 4
	jne .LBB214_136
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
	mov rdi, rbx
	call _Unwind_Resume@PLT
.LBB214_136:
	add r12d, 512
.Ltmp23445:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	mov rdi, rbx
	call _Unwind_Resume@PLT
.Ltmp23446:
.Ltmp22972:
	mov rdi, rax
.Ltmp23447:
	add r12d, 512
.Ltmp23448:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	call _Unwind_Resume@PLT
.Ltmp23449:
.Ltmp22969:
	mov rdi, rax
	jmp .LBB214_143
.Ltmp22964:
	mov rdi, rax
	mov r12d, r14d
.Ltmp23451:
.LBB214_140:
	test r12b, 4
	jne .LBB214_142
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
	jmp .LBB214_143
.LBB214_142:
	add r12d, 512
	and r12d, -1342177792
	mov dword ptr [r15], r12d
.Ltmp23454:
.LBB214_143:
	mov eax, dword ptr [rsp + 64]
	mov rcx, qword ptr [rsp + 80]
.Ltmp23455:
	add eax, 512
	and eax, -1342177792
	mov dword ptr [rcx], eax
	call _Unwind_Resume@PLT
.Ltmp23456:
.Lfunc_end214:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic, .Lfunc_end214-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic","a",@progbits
	.p2align	2, 0x0
GCC_except_table214:
masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic:
.Lfunc_begin222:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception93
	push rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov rbp, rsp
	.cfi_def_cfa_register rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	and rsp, -64
	sub rsp, 832
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	mov rbx, r9
	mov r12d, ecx
	mov r15, rdx
	mov r13, rsi
	mov qword ptr [rsp + 72], rdi
.Ltmp25183:
.Ltmp25159:
	lea rdi, [rsp + 192]
.Ltmp25184:
	mov rdx, r8
.Ltmp25185:
	call <masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point
.Ltmp25186:
.Ltmp25160:
	cmp dword ptr [rsp + 192], 1
	jne .LBB222_13
	mov rax, qword ptr [rsp + 200]
	mov qword ptr [rsp + 88], rax
.Ltmp25189:
	mov ecx, dword ptr [r13]
	mov dword ptr [rsp + 44], 0
.Ltmp25190:
	mov rax, qword ptr [r13 + 560]
.Ltmp25191:
	mov dword ptr [rsp + 68], 0
.Ltmp25192:
	test ecx, 1073741824
.Ltmp25193:
	je .LBB222_4
.Ltmp25194:
	test rax, rax
	sete al
.Ltmp25195:
	mov rcx, qword ptr [rsp + 72]
.Ltmp25196:
	mov rcx, qword ptr [rcx + 992]
.Ltmp25197:
	cmp rcx, r13
	sete cl
.Ltmp25198:
	mov dword ptr [rsp + 68], ecx
.Ltmp25199:
	setne cl
.Ltmp25200:
	and cl, al
	mov dword ptr [rsp + 44], ecx
.Ltmp25201:
.LBB222_4:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 229], xmm0
	movups xmmword ptr [rsp + 213], xmm0
	movups xmmword ptr [rsp + 197], xmm0
	mov qword ptr [rsp + 244], 0
	lea rdi, [rsp + 272]
	mov edx, 456
	xor esi, esi
	call qword ptr [rip + memset@GOTPCREL]
	mov dword ptr [rsp + 192], -2147483648
	mov byte ptr [rsp + 196], 0
	movabs rax, 2399420405009046
	mov qword ptr [rsp + 264], rax
	movabs rax, -4127003096615560480
	mov qword ptr [rsp + 256], rax
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 728], xmm0
	movups xmmword ptr [rsp + 744], xmm0
.Ltmp25202:
	mov edi, 576
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25203:
	test rax, rax
	je .LBB222_117
.Ltmp25204:
	mov r14, rax
	lea rsi, [rsp + 192]
	mov edx, 576
	mov rdi, rax
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp25205:
	or dword ptr [r15], 4
.Ltmp25206:
	#MEMBARRIER
.Ltmp25162:
	lea rdi, [rsp + 192]
	mov qword ptr [rsp + 168], rdi
.Ltmp25207:
	mov rsi, r13
	mov rdx, qword ptr [rsp + 88]
	mov rcx, r14
	mov r8, rbx
	call <masstree::leaf24::LeafNode24<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated
.Ltmp25163:
	mov r14, qword ptr [rsp + 192]
.Ltmp25209:
	mov rax, qword ptr [rsp + 200]
.Ltmp25210:
	mov qword ptr [rsp + 80], rax
.Ltmp25211:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25212:
	test rax, rax
	mov ebx, dword ptr [rsp + 68]
.Ltmp25213:
	je .LBB222_118
.Ltmp25214:
	mov rcx, rax
	mov qword ptr [rax], r14
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 72]
.Ltmp25216:
	mov rax, qword ptr [rdx + 960]
.Ltmp25217:
	mov qword ptr [rcx + 8], rax
.Ltmp25218:
	lock cmpxchg	qword ptr [rdx + 960], rcx
.Ltmp25219:
	je .LBB222_9
.Ltmp25220:
	.p2align	4
.LBB222_8:
	pause
.Ltmp25222:
	mov rax, qword ptr [rdx + 960]
.Ltmp25223:
	mov qword ptr [rcx + 8], rax
.Ltmp25224:
	lock cmpxchg	qword ptr [rdx + 960], rcx
.Ltmp25225:
	jne .LBB222_8
.Ltmp25226:
.LBB222_9:
	lock inc	qword ptr [rdx + 968]
.Ltmp25227:
	mov rcx, qword ptr [r13 + 544]
.Ltmp25228:
	test cl, 1
	je .LBB222_11
	jmp .LBB222_10
.Ltmp25229:
	.p2align	4
.LBB222_12:
	pause
.Ltmp25231:
	mov rcx, qword ptr [r13 + 544]
.Ltmp25232:
	test cl, 1
	je .LBB222_11
.Ltmp25233:
.LBB222_10:
	mov rdi, r13
	call masstree::leaf24::LeafNode24<S>::wait_for_split
.Ltmp25234:
	mov rcx, qword ptr [r13 + 544]
.Ltmp25235:
	test cl, 1
	jne .LBB222_10
.Ltmp25236:
.LBB222_11:
	lea rdx, [rcx + 1]
.Ltmp25237:
	mov rax, rcx
	lock cmpxchg	qword ptr [r13 + 544], rdx
.Ltmp25238:
	jne .LBB222_12
.Ltmp25239:
	mov qword ptr [r14 + 552], r13
.Ltmp25240:
	mov qword ptr [r14 + 544], rcx
.Ltmp25241:
	test rcx, rcx
.Ltmp25242:
	je .LBB222_17
.Ltmp25243:
	mov qword ptr [rcx + 552], r14
.Ltmp25244:
.LBB222_17:
	#MEMBARRIER
	mov qword ptr [r13 + 544], r14
	mov byte ptr [rsp + 38], bl
	mov eax, dword ptr [rsp + 44]
	mov byte ptr [rsp + 39], al
	mov cl, 1
	mov eax, 1
	mov esi, 560
	xorps xmm0, xmm0
	mov r9, qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	mov r10d, -268435464
	jmp .LBB222_19
.Ltmp25246:
	.p2align	4
.LBB222_18:
	test dl, 1
	mov rax, r14
	cmovne rax, r15
.Ltmp25247:
	mov qword ptr [r9 + r11], rax
.Ltmp25248:
	mov eax, dword ptr [r9]
.Ltmp25249:
	add eax, 512
	and eax, -1342177792
.Ltmp25250:
	mov ecx, dword ptr [rsp + 40]
.Ltmp25251:
	add ecx, 512
	and ecx, -1342177792
.Ltmp25252:
	cmp qword ptr [rsp + 176], r15
.Ltmp25253:
	#MEMBARRIER
	mov dword ptr [r9], eax
	mov rax, qword ptr [rsp + 56]
.Ltmp25254:
	mov dword ptr [rax], ecx
.Ltmp25255:
	sete byte ptr [rsp + 38]
	mov eax, dword ptr [rsp + 44]
	mov byte ptr [rsp + 39], al
.Ltmp25256:
	mov rdx, qword ptr [rsp + 88]
.Ltmp25257:
	lea rax, [rdx + 1]
	xor ecx, ecx
	mov r13, r15
	cmp rdx, 63
	mov esi, 560
	mov r9, qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	ja .LBB222_121
.Ltmp25258:
.LBB222_19:
	mov r8, r14
.Ltmp25259:
	mov dword ptr [rsp + 40], r12d
.Ltmp25260:
	test cl, 1
	mov r11d, 264
	cmovne r11, rsi
	mov qword ptr [rsp + 56], r15
.Ltmp25261:
	not bl
	mov rsi, rax
	xor edx, edx
	mov qword ptr [rsp + 48], r14
.Ltmp25262:
	jmp .LBB222_23
.Ltmp25263:
	.p2align	4
.LBB222_20:
	mov rax, qword ptr [r13 + r11]
.Ltmp25264:
	cmp rax, r15
	je .LBB222_34
	xor edx, edx
.Ltmp25266:
.LBB222_22:
	lea eax, [r12 + 512]
	mov esi, r12d
	add esi, 8
	test r12b, 4
	mov edi, -1342177792
	cmove edi, r10d
	cmovne esi, eax
	and esi, edi
	mov dword ptr [r15], esi
.Ltmp25267:
	pause
.Ltmp25268:
	mov rax, qword ptr [rsp + 88]
.Ltmp25269:
	cmp rax, 63
	lea rsi, [rax + 1]
.Ltmp25270:
	ja .LBB222_120
.Ltmp25271:
.LBB222_23:
	mov r15, qword ptr [r13 + r11]
.Ltmp25272:
	test r15, r15
.Ltmp25273:
	jne .LBB222_25
.Ltmp25274:
	test byte ptr [rsp + 44], 1
	jne .LBB222_73
.Ltmp25275:
.LBB222_25:
	test r15, r15
	setne al
.Ltmp25276:
	or al, bl
	test al, 1
	je .LBB222_79
.Ltmp25277:
	test r15, r15
.Ltmp25278:
	je .LBB222_119
.Ltmp25279:
	mov qword ptr [rsp + 88], rsi
	xor esi, esi
	jmp .LBB222_29
.Ltmp25280:
	.p2align	4
.LBB222_28:
	and esi, 7
	lea esi, [2*rsi + 1]
.LBB222_29:
	mov r12d, dword ptr [r15]
	test r12b, 1
	jne .LBB222_31
	mov r14d, r12d
	or r14d, 3
	mov eax, r12d
	lock cmpxchg	dword ptr [r15], r14d
	je .LBB222_20
.LBB222_31:
	xor eax, eax
	.p2align	4
.LBB222_32:
	mov edi, eax
	pause
	cmp eax, esi
	adc eax, 0
	cmp edi, esi
	jae .LBB222_28
	cmp eax, esi
	jbe .LBB222_32
	jmp .LBB222_28
.Ltmp25286:
	.p2align	4
.LBB222_34:
	mov edi, ebx
.Ltmp25288:
	movzx eax, byte ptr [r15 + 4]
.Ltmp25289:
	movzx ebx, al
.Ltmp25290:
	test bl, bl
.Ltmp25291:
	je .LBB222_41
	xor eax, eax
	jmp .LBB222_36
.Ltmp25293:
	.p2align	4
.LBB222_38:
	mov rsi, qword ptr [r15 + 256]
.Ltmp25294:
	cmp rsi, r13
.Ltmp25295:
	je .LBB222_44
.Ltmp25296:
.LBB222_39:
	inc rax
.Ltmp25297:
	cmp rbx, rax
.Ltmp25298:
	je .LBB222_40
.Ltmp25299:
.LBB222_36:
	cmp rax, 15
	jae .LBB222_38
.Ltmp25300:
	mov rsi, qword ptr [r15 + 8*rax + 136]
.Ltmp25301:
	cmp rsi, r13
.Ltmp25302:
	jne .LBB222_39
	jmp .LBB222_44
.Ltmp25303:
.LBB222_40:
	dec rax
.Ltmp25304:
	cmp rax, 14
	jae .LBB222_42
.Ltmp25305:
.LBB222_41:
	mov rax, qword ptr [r15 + 8*rbx + 136]
.Ltmp25306:
	cmp rax, r13
.Ltmp25307:
	jne .LBB222_43
	jmp .LBB222_45
.Ltmp25308:
.LBB222_42:
	mov rax, qword ptr [r15 + 256]
.Ltmp25309:
	cmp rax, r13
.Ltmp25310:
	je .LBB222_45
.Ltmp25311:
.LBB222_43:
	inc rdx
	cmp rdx, 16
	mov ebx, edi
	jbe .LBB222_22
	jmp .LBB222_102
	.p2align	4
.LBB222_44:
	mov rbx, rax
.Ltmp25315:
.LBB222_45:
	movzx eax, byte ptr [r15 + 4]
.Ltmp25316:
	cmp al, 14
	mov rax, qword ptr [rsp + 72]
.Ltmp25317:
	jbe .LBB222_103
.Ltmp25318:
	or dword ptr [r15], 4
.Ltmp25319:
	#MEMBARRIER
	mov rcx, qword ptr [rax + 992]
.Ltmp25320:
	cmp rcx, r15
	sete al
	mov dword ptr [rsp + 68], eax
.Ltmp25321:
	mov qword ptr [rsp + 184], r11
	mov qword ptr [rsp + 176], rcx
.Ltmp25322:
	je .LBB222_49
.Ltmp25323:
	mov rax, qword ptr [r15 + 264]
.Ltmp25324:
	test rax, rax
.Ltmp25325:
	je .LBB222_50
.LBB222_49:
	mov dword ptr [rsp + 44], 0
	jmp .LBB222_51
.Ltmp25327:
.LBB222_50:
	mov eax, dword ptr [r15]
.Ltmp25328:
	shr eax, 30
.Ltmp25329:
	and al, 1
	mov dword ptr [rsp + 44], eax
.Ltmp25330:
.LBB222_51:
	or r12d, 7
.Ltmp25331:
	mov eax, dword ptr [r15 + 8]
	mov dword ptr [rsp + 116], eax
.Ltmp25333:
	mov r13d, dword ptr [r15]
.Ltmp25334:
	lea rax, [rsp + 196]
.Ltmp25335:
	movups xmmword ptr [rax + 96], xmm0
	movups xmmword ptr [rax + 80], xmm0
	movups xmmword ptr [rax + 64], xmm0
	movups xmmword ptr [rax + 48], xmm0
	movups xmmword ptr [rax + 32], xmm0
	movups xmmword ptr [rax + 16], xmm0
	movups xmmword ptr [rax], xmm0
	mov qword ptr [rax + 112], 0
.Ltmp25336:
	mov eax, 320
	mov qword ptr [rsp + 160], rax
	mov eax, 64
	mov qword ptr [rsp + 152], rax
.Ltmp25337:
	mov edi, 320
	mov esi, 64
	call r9
.Ltmp25338:
	test rax, rax
	je .LBB222_115
.Ltmp25339:
	mov r14, rax
	and r13d, -2147483648
	or r13d, 5
.Ltmp25340:
	mov dword ptr [rax], r13d
	mov byte ptr [rax + 4], 0
	mov eax, dword ptr [rsp + 116]
	mov dword ptr [r14 + 8], eax
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [r14 + 12], xmm0
	movups xmmword ptr [r14 + 28], xmm1
	movups xmmword ptr [r14 + 44], xmm2
	movups xmmword ptr [r14 + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [r14 + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [r14 + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [r14 + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [r14 + 120], xmm0
	xorps xmm0, xmm0
	movups xmmword ptr [r14 + 136], xmm0
	movups xmmword ptr [r14 + 152], xmm0
	movups xmmword ptr [r14 + 168], xmm0
	movups xmmword ptr [r14 + 184], xmm0
	movups xmmword ptr [r14 + 200], xmm0
	movups xmmword ptr [r14 + 216], xmm0
	movups xmmword ptr [r14 + 232], xmm0
	movups xmmword ptr [r14 + 248], xmm0
	mov qword ptr [r14 + 264], 0
.Ltmp25341:
	mov eax, 16
	mov qword ptr [rsp + 160], rax
	mov eax, 8
	mov qword ptr [rsp + 152], rax
.Ltmp25342:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25343:
	test rax, rax
	je .LBB222_115
.Ltmp25344:
	mov rcx, rax
	mov qword ptr [rax], r14
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 72]
.Ltmp25346:
	mov rax, qword ptr [rdx + 976]
.Ltmp25347:
	mov qword ptr [rcx + 8], rax
.Ltmp25348:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25349:
	je .LBB222_55
.Ltmp25350:
	.p2align	4
.LBB222_54:
	pause
.Ltmp25352:
	mov rax, qword ptr [rdx + 976]
.Ltmp25353:
	mov qword ptr [rcx + 8], rax
.Ltmp25354:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25355:
	jne .LBB222_54
.Ltmp25356:
.LBB222_55:
	lock inc	qword ptr [rdx + 984]
.Ltmp25357:
.Ltmp25166:
	mov rdi, r15
	mov rsi, r14
	mov rdx, r14
	mov rcx, rbx
	mov r8, qword ptr [rsp + 80]
	mov r9, qword ptr [rsp + 48]
	call masstree::internode::InternodeNode<S,_>::split_into
.Ltmp25358:
.Ltmp25167:
	mov qword ptr [rsp + 80], rax
.Ltmp25359:
	lea rax, [r14 + 136]
.Ltmp25360:
	cmp dword ptr [r15 + 8], 0
.Ltmp25361:
	movzx ecx, byte ptr [r14 + 4]
.Ltmp25362:
	movzx ecx, cl
	mov ebx, dword ptr [rsp + 68]
.Ltmp25363:
	xorps xmm0, xmm0
	mov r10d, -268435464
	mov r11, qword ptr [rsp + 184]
.Ltmp25364:
	je .LBB222_64
	xor edi, edi
.Ltmp25366:
	xor esi, esi
	mov r9, qword ptr [rsp + 48]
.Ltmp25367:
	.p2align	4
.LBB222_58:
	cmp rdi, rcx
.Ltmp25369:
	adc rsi, 0
.Ltmp25370:
	cmp rdi, 15
	jae .LBB222_60
.Ltmp25371:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp25372:
	test r8, r8
.Ltmp25373:
	jne .LBB222_61
	jmp .LBB222_62
.Ltmp25374:
	.p2align	4
.LBB222_60:
	mov r8, qword ptr [r14 + 256]
.Ltmp25376:
	test r8, r8
.Ltmp25377:
	je .LBB222_62
.Ltmp25378:
.LBB222_61:
	mov qword ptr [r8 + 264], r14
.Ltmp25379:
.LBB222_62:
	cmp rdi, rcx
.Ltmp25380:
	jae .LBB222_18
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB222_58
	jmp .LBB222_18
.Ltmp25382:
	.p2align	4
.LBB222_64:
	xor edi, edi
.Ltmp25384:
	xor esi, esi
	mov r9, qword ptr [rsp + 48]
.Ltmp25385:
	.p2align	4
.LBB222_65:
	cmp rdi, rcx
.Ltmp25387:
	adc rsi, 0
.Ltmp25388:
	cmp rdi, 15
	jae .LBB222_67
.Ltmp25389:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp25390:
	test r8, r8
.Ltmp25391:
	jne .LBB222_68
	jmp .LBB222_69
.Ltmp25392:
	.p2align	4
.LBB222_67:
	mov r8, qword ptr [r14 + 256]
.Ltmp25394:
	test r8, r8
.Ltmp25395:
	je .LBB222_69
.Ltmp25396:
.LBB222_68:
	mov qword ptr [r8 + 560], r14
.Ltmp25397:
.LBB222_69:
	cmp rdi, rcx
.Ltmp25398:
	jae .LBB222_18
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB222_65
	jmp .LBB222_18
.Ltmp25400:
.LBB222_13:
	test r12b, 4
	jne .LBB222_71
	mov eax, r12d
	and eax, 2
	lea r12d, [r12 + 4*rax]
.Ltmp25402:
	and r12d, -268435464
	jmp .LBB222_72
.Ltmp25403:
.LBB222_71:
	add r12d, 512
.Ltmp25404:
	and r12d, -1342177792
.LBB222_72:
	mov dword ptr [r15], r12d
	mov al, 4
.Ltmp25406:
	jmp .LBB222_101
.Ltmp25407:
.LBB222_73:
	test cl, 1
	je .LBB222_86
.Ltmp25408:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp25409:
	mov ebx, 320
	mov r14d, 64
.Ltmp25410:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25411:
	test rax, rax
	je .LBB222_116
.Ltmp25412:
	mov r15, rax
.Ltmp25413:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp25414:
	xorps xmm0, xmm0
.Ltmp25415:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp25416:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp25418:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 48]
.Ltmp25420:
	mov qword ptr [r15 + 144], rax
.Ltmp25421:
	mov byte ptr [r15 + 4], 1
.Ltmp25422:
	lock or	dword ptr [r15], 1073741824
.Ltmp25423:
	mov ebx, 16
	mov r14d, 8
.Ltmp25424:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25425:
	test rax, rax
	je .LBB222_116
.Ltmp25426:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 72]
.Ltmp25428:
	mov rax, qword ptr [rdx + 976]
.Ltmp25429:
	mov qword ptr [rcx + 8], rax
.Ltmp25430:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25431:
	je .LBB222_78
.Ltmp25432:
	.p2align	4
.LBB222_77:
	pause
.Ltmp25434:
	mov rax, qword ptr [rdx + 976]
.Ltmp25435:
	mov qword ptr [rcx + 8], rax
.Ltmp25436:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25437:
	jne .LBB222_77
.Ltmp25438:
.LBB222_78:
	lock inc	qword ptr [rdx + 984]
.Ltmp25439:
	#MEMBARRIER
	mov qword ptr [r13 + 560], r15
	mov rcx, qword ptr [rsp + 48]
.Ltmp25441:
	mov qword ptr [rcx + 560], r15
.Ltmp25442:
	lock and	dword ptr [r13], -1073741825
	mov r13, rcx
.Ltmp25444:
	jmp .LBB222_98
.Ltmp25445:
.LBB222_79:
	test cl, 1
	je .LBB222_91
.Ltmp25446:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp25447:
	mov ebx, 320
	mov r14d, 64
.Ltmp25448:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25449:
	test rax, rax
	je .LBB222_116
.Ltmp25450:
	mov r15, rax
.Ltmp25451:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp25452:
	xorps xmm0, xmm0
.Ltmp25453:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp25454:
	lock or	dword ptr [rax], 1073741824
.Ltmp25455:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp25457:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 48]
.Ltmp25459:
	mov qword ptr [r15 + 144], rax
.Ltmp25460:
	mov byte ptr [r15 + 4], 1
.Ltmp25461:
	mov ebx, 16
	mov r14d, 8
.Ltmp25462:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25463:
	test rax, rax
	je .LBB222_116
.Ltmp25464:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 72]
.Ltmp25466:
	mov rax, qword ptr [rdx + 976]
.Ltmp25467:
	mov qword ptr [rcx + 8], rax
.Ltmp25468:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25469:
	je .LBB222_84
.Ltmp25470:
	.p2align	4
.LBB222_83:
	pause
.Ltmp25472:
	mov rax, qword ptr [rdx + 976]
.Ltmp25473:
	mov qword ptr [rcx + 8], rax
.Ltmp25474:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25475:
	jne .LBB222_83
.Ltmp25476:
.LBB222_84:
	lock inc	qword ptr [rdx + 984]
.Ltmp25477:
	mov qword ptr [rsp + 96], r13
.Ltmp25478:
	mov rax, r13
	lock cmpxchg	qword ptr [rdx + 992], r15
	mov rcx, qword ptr [rsp + 48]
.Ltmp25479:
	jne .LBB222_126
.Ltmp25480:
	#MEMBARRIER
	mov qword ptr [r13 + 560], r15
.Ltmp25481:
	mov qword ptr [rcx + 560], r15
.Ltmp25482:
	jmp .LBB222_98
.Ltmp25483:
.LBB222_86:
	mov r12d, dword ptr [r13 + 8]
.Ltmp25484:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp25485:
	mov ebx, 320
	mov r14d, 64
.Ltmp25486:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25487:
	test rax, rax
	je .LBB222_116
.Ltmp25488:
	mov r15, rax
.Ltmp25489:
	inc r12d
.Ltmp25490:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r12d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp25491:
	xorps xmm0, xmm0
.Ltmp25492:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp25493:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp25495:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 48]
.Ltmp25497:
	mov qword ptr [r15 + 144], rax
.Ltmp25498:
	mov byte ptr [r15 + 4], 1
.Ltmp25499:
	lock or	dword ptr [r15], 1073741824
.Ltmp25500:
	mov ebx, 16
	mov r14d, 8
.Ltmp25501:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25502:
	test rax, rax
	je .LBB222_116
.Ltmp25503:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 72]
.Ltmp25505:
	mov rax, qword ptr [rdx + 976]
.Ltmp25506:
	mov qword ptr [rcx + 8], rax
.Ltmp25507:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25508:
	je .LBB222_90
.Ltmp25509:
	.p2align	4
.LBB222_89:
	pause
.Ltmp25511:
	mov rax, qword ptr [rdx + 976]
.Ltmp25512:
	mov qword ptr [rcx + 8], rax
.Ltmp25513:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25514:
	jne .LBB222_89
.Ltmp25515:
.LBB222_90:
	lock inc	qword ptr [rdx + 984]
.Ltmp25516:
	#MEMBARRIER
	mov qword ptr [r13 + 264], r15
	mov rcx, qword ptr [rsp + 48]
.Ltmp25518:
	jmp .LBB222_97
.Ltmp25519:
.LBB222_91:
	mov r12d, dword ptr [r13 + 8]
.Ltmp25520:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp25521:
	mov ebx, 320
	mov r14d, 64
.Ltmp25522:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25523:
	test rax, rax
	je .LBB222_116
.Ltmp25524:
	mov r15, rax
.Ltmp25525:
	inc r12d
.Ltmp25526:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r12d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp25527:
	xorps xmm0, xmm0
.Ltmp25528:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp25529:
	lock or	dword ptr [rax], 1073741824
.Ltmp25530:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 80]
.Ltmp25532:
	mov qword ptr [r15 + 16], rax
	mov rax, qword ptr [rsp + 48]
.Ltmp25534:
	mov qword ptr [r15 + 144], rax
.Ltmp25535:
	mov byte ptr [r15 + 4], 1
.Ltmp25536:
	mov ebx, 16
	mov r14d, 8
.Ltmp25537:
	mov edi, 16
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp25538:
	test rax, rax
	je .LBB222_116
.Ltmp25539:
	mov rcx, rax
	mov qword ptr [rax], r15
	mov qword ptr [rax + 8], 0
	mov rdx, qword ptr [rsp + 72]
.Ltmp25541:
	mov rax, qword ptr [rdx + 976]
.Ltmp25542:
	mov qword ptr [rcx + 8], rax
.Ltmp25543:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25544:
	je .LBB222_95
.Ltmp25545:
	.p2align	4
.LBB222_94:
	pause
.Ltmp25547:
	mov rax, qword ptr [rdx + 976]
.Ltmp25548:
	mov qword ptr [rcx + 8], rax
.Ltmp25549:
	lock cmpxchg	qword ptr [rdx + 976], rcx
.Ltmp25550:
	jne .LBB222_94
.Ltmp25551:
.LBB222_95:
	lock inc	qword ptr [rdx + 984]
.Ltmp25552:
	mov qword ptr [rsp + 96], r13
.Ltmp25553:
	mov rax, r13
	lock cmpxchg	qword ptr [rdx + 992], r15
	mov rcx, qword ptr [rsp + 48]
.Ltmp25554:
	jne .LBB222_127
.Ltmp25555:
	#MEMBARRIER
	mov qword ptr [r13 + 264], r15
.Ltmp25556:
.LBB222_97:
	mov qword ptr [rcx + 264], r15
.Ltmp25557:
.LBB222_98:
	lock and	dword ptr [r13], -1073741825
.Ltmp25558:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
.Ltmp25559:
.LBB222_99:
	mov al, 6
.LBB222_100:
	mov ecx, dword ptr [rsp + 40]
	mov rdx, qword ptr [rsp + 56]
.Ltmp25561:
	add ecx, 512
	and ecx, -1342177792
	mov dword ptr [rdx], ecx
.Ltmp25562:
.LBB222_101:
	lea rsp, [rbp - 40]
.Ltmp25563:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	.cfi_def_cfa rsp, 8
	ret
.Ltmp25564:
.LBB222_102:
	.cfi_def_cfa rbp, 16
	mov eax, dword ptr [r8]
	add eax, 512
	and eax, -1342177792
.Ltmp25565:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
.Ltmp25566:
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp25567:
	#MEMBARRIER
	mov dword ptr [r8], eax
.Ltmp25568:
	and ecx, esi
	mov dword ptr [r15], ecx
	mov al, 4
	jmp .LBB222_100
.Ltmp25570:
.LBB222_103:
	movzx eax, byte ptr [r15 + 4]
.Ltmp25571:
	movzx eax, al
.Ltmp25572:
	cmp rbx, rax
.Ltmp25573:
	jae .LBB222_110
.Ltmp25574:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.206]
	mov rcx, rax
	jmp .LBB222_106
.Ltmp25575:
	.p2align	4
.LBB222_105:
	mov qword ptr [r15 + 8*rsi + 152], rdi
	mov rcx, rsi
.Ltmp25577:
	cmp rbx, rsi
.Ltmp25578:
	jae .LBB222_110
.Ltmp25579:
.LBB222_106:
	cmp rcx, 16
	jae .LBB222_123
.Ltmp25580:
	lea rsi, [rcx - 1]
.Ltmp25581:
	mov rdi, qword ptr [r15 + 8*rsi + 16]
.Ltmp25582:
	cmp rcx, 15
	je .LBB222_124
.Ltmp25583:
	mov qword ptr [r15 + 8*rsi + 24], rdi
.Ltmp25584:
	mov rdi, qword ptr [r15 + 8*rsi + 144]
.Ltmp25585:
	cmp rcx, 14
	jb .LBB222_105
.Ltmp25586:
	mov qword ptr [r15 + 256], rdi
	mov rcx, rsi
.Ltmp25588:
	cmp rbx, rsi
.Ltmp25589:
	jb .LBB222_106
.Ltmp25590:
.LBB222_110:
	cmp rbx, 14
	ja .LBB222_130
	mov rcx, qword ptr [rsp + 80]
.Ltmp25592:
	mov qword ptr [r15 + 8*rbx + 16], rcx
.Ltmp25593:
	jne .LBB222_113
.Ltmp25594:
	mov qword ptr [r15 + 256], r8
.Ltmp25595:
	jmp .LBB222_114
.Ltmp25596:
.LBB222_113:
	mov qword ptr [r15 + 8*rbx + 144], r8
.Ltmp25597:
.LBB222_114:
	#MEMBARRIER
	inc al
.Ltmp25598:
	mov byte ptr [r15 + 4], al
.Ltmp25599:
	mov qword ptr [r8 + r11], r15
.Ltmp25600:
	mov eax, dword ptr [r8]
	add eax, 512
	and eax, -1342177792
.Ltmp25602:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp25603:
	#MEMBARRIER
	mov dword ptr [r8], eax
.Ltmp25604:
	and ecx, esi
	mov dword ptr [r15], ecx
	jmp .LBB222_99
.Ltmp25605:
.LBB222_115:
.Ltmp25169:
	mov r14d, r12d
	mov rdi, qword ptr [rsp + 152]
	mov rsi, qword ptr [rsp + 160]
.Ltmp25606:
	call alloc::alloc::handle_alloc_error
.Ltmp25170:
	jmp .LBB222_129
.Ltmp25607:
.LBB222_116:
.Ltmp25174:
	mov rdi, r14
	mov rsi, rbx
	call alloc::alloc::handle_alloc_error
.Ltmp25175:
	jmp .LBB222_129
.Ltmp25608:
.LBB222_117:
.Ltmp25180:
	mov edi, 64
	mov esi, 576
	call alloc::alloc::handle_alloc_error
.Ltmp25181:
	jmp .LBB222_129
.Ltmp25609:
.LBB222_118:
.Ltmp25177:
	mov edi, 8
	mov esi, 16
	call alloc::alloc::handle_alloc_error
.Ltmp25178:
	jmp .LBB222_129
.Ltmp25610:
.LBB222_119:
	mov eax, dword ptr [r8]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [r8], eax
.Ltmp25611:
	mov eax, dword ptr [rsp + 40]
.Ltmp25612:
	add eax, 512
	and eax, -1342177792
	mov rcx, qword ptr [rsp + 56]
	mov dword ptr [rcx], eax
	lea rax, [rsp + 38]
.Ltmp25614:
	mov qword ptr [rsp + 120], rax
	lea rcx, [rip + <bool as core::fmt::Display>::fmt]
	mov qword ptr [rsp + 128], rcx
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.166]
	mov eax, 2
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.164]
.Ltmp25615:
	lea r8, [rsp + 120]
	mov edi, 24
	lea r10, [rsp + 39]
	lea r9, [rsp + 136]
	jmp .LBB222_122
.Ltmp25616:
.LBB222_120:
	mov r15, qword ptr [rsp + 56]
	mov eax, dword ptr [rsp + 40]
	mov r12d, eax
	mov r14, r8
.Ltmp25617:
.LBB222_121:
	mov eax, dword ptr [r14]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [r14], eax
.Ltmp25618:
	add r12d, 512
.Ltmp25619:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.171]
	mov eax, 1
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.170]
	lea rcx, [rip + core::fmt::num::imp::<impl core::fmt::Display for u64>::fmt]
	lea r9, [rsp + 120]
	mov edi, 8
	lea r10, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.167]
	mov r8, r9
.Ltmp25620:
.LBB222_122:
	mov qword ptr [r9], r10
	mov qword ptr [r8 + rdi], rcx
	mov rdi, qword ptr [rsp + 168]
	mov qword ptr [rdi], rdx
	mov qword ptr [rdi + 8], 2
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], r8
	mov qword ptr [rdi + 24], rax
	call core::panicking::panic_fmt
.Ltmp25621:
.LBB222_123:
	dec rcx
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.205]
	mov rbx, rcx
.Ltmp25622:
	jmp .LBB222_125
.Ltmp25623:
.LBB222_124:
	mov ebx, 15
.Ltmp25624:
.LBB222_125:
.Ltmp25164:
	mov esi, 15
	mov rdi, rbx
	call core::panicking::panic_bounds_check
.Ltmp25625:
.Ltmp25165:
	jmp .LBB222_129
.Ltmp25626:
.LBB222_126:
	mov qword ptr [rsp + 104], rax
	lea rax, [rsp + 96]
.Ltmp25628:
	mov qword ptr [rsp + 120], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 128], rax
	lea rcx, [rsp + 104]
	mov qword ptr [rsp + 136], rcx
	mov qword ptr [rsp + 144], rax
.Ltmp25629:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.175]
.Ltmp25630:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.177]
	jmp .LBB222_128
.Ltmp25631:
.LBB222_127:
	mov qword ptr [rsp + 104], rax
	lea rax, [rsp + 96]
.Ltmp25633:
	mov qword ptr [rsp + 120], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 128], rax
	lea rcx, [rsp + 104]
	mov qword ptr [rsp + 136], rcx
	mov qword ptr [rsp + 144], rax
.Ltmp25634:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.180]
.Ltmp25635:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.181]
.Ltmp25636:
.LBB222_128:
	lea rax, [rsp + 120]
	lea rdi, [rsp + 192]
	mov qword ptr [rdi + 8], 3
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], rax
	mov qword ptr [rdi + 24], 2
.Ltmp25172:
	call core::panicking::panic_fmt
.Ltmp25637:
.Ltmp25173:
.LBB222_129:
	ud2
.Ltmp25638:
.LBB222_130:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.204]
.Ltmp25639:
	jmp .LBB222_125
.Ltmp25640:
.Ltmp25168:
	mov rdi, rax
	jmp .LBB222_140
.Ltmp25641:
.Ltmp25161:
	mov rbx, rax
.Ltmp25642:
	jmp .LBB222_134
.Ltmp25643:
.Ltmp25182:
	mov rbx, rax
.Ltmp25644:
	lea rdi, [rsp + 192]
.Ltmp25645:
	call core::ptr::drop_in_place<masstree::leaf24::LeafNode24<masstree::value::LeafValue<u64>>>
.Ltmp25646:
.LBB222_134:
	test r12b, 4
	jne .LBB222_136
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
	mov rdi, rbx
	call _Unwind_Resume@PLT
.LBB222_136:
	add r12d, 512
.Ltmp25649:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	mov rdi, rbx
	call _Unwind_Resume@PLT
.Ltmp25650:
.Ltmp25179:
	mov rdi, rax
.Ltmp25651:
	add r12d, 512
.Ltmp25652:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	call _Unwind_Resume@PLT
.Ltmp25653:
.Ltmp25176:
	mov rdi, rax
	jmp .LBB222_143
.Ltmp25171:
	mov rdi, rax
	mov r12d, r14d
.Ltmp25655:
.LBB222_140:
	test r12b, 4
	jne .LBB222_142
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
	jmp .LBB222_143
.LBB222_142:
	add r12d, 512
	and r12d, -1342177792
	mov dword ptr [r15], r12d
.Ltmp25658:
.LBB222_143:
	mov eax, dword ptr [rsp + 40]
	mov rcx, qword ptr [rsp + 56]
.Ltmp25659:
	add eax, 512
	and eax, -1342177792
	mov dword ptr [rcx], eax
	call _Unwind_Resume@PLT
.Ltmp25660:
.Lfunc_end222:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic, .Lfunc_end222-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic","a",@progbits
	.p2align	2, 0x0
GCC_except_table222:
masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic:
.Lfunc_begin226:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception97
	push rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov rbp, rsp
	.cfi_def_cfa_register rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	and rsp, -64
	sub rsp, 704
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	mov rbx, r9
	mov r12d, ecx
	mov r15, rdx
	mov r13, rsi
	mov qword ptr [rsp + 72], rdi
.Ltmp26862:
.Ltmp26780:
	lea rdi, [rsp + 192]
.Ltmp26863:
	mov rdx, r8
.Ltmp26864:
	call <masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::calculate_split_point
.Ltmp26865:
.Ltmp26781:
	cmp dword ptr [rsp + 192], 1
	jne .LBB226_15
	mov rax, qword ptr [rsp + 200]
	mov qword ptr [rsp + 80], rax
.Ltmp26868:
	mov ecx, dword ptr [r13]
	mov dword ptr [rsp + 68], 0
.Ltmp26869:
	mov rax, qword ptr [r13 + 344]
.Ltmp26870:
	mov edx, 0
.Ltmp26871:
	test ecx, 1073741824
.Ltmp26872:
	je .LBB226_4
.Ltmp26873:
	test rax, rax
	sete al
.Ltmp26874:
	mov rcx, qword ptr [rsp + 72]
.Ltmp26875:
	mov rcx, qword ptr [rcx + 1024]
.Ltmp26876:
	cmp rcx, r13
	sete dl
.Ltmp26877:
	setne cl
.Ltmp26878:
	and cl, al
	mov dword ptr [rsp + 68], ecx
.Ltmp26879:
.LBB226_4:
	mov dword ptr [rsp + 108], edx
.Ltmp26880:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 213], xmm0
	movups xmmword ptr [rsp + 197], xmm0
.Ltmp26881:
	movups xmmword ptr [rsp + 392], xmm0
	movups xmmword ptr [rsp + 408], xmm0
	movups xmmword ptr [rsp + 424], xmm0
	movups xmmword ptr [rsp + 440], xmm0
	movups xmmword ptr [rsp + 456], xmm0
	movups xmmword ptr [rsp + 472], xmm0
	movups xmmword ptr [rsp + 488], xmm0
	mov qword ptr [rsp + 504], 0
.Ltmp26882:
	movaps xmmword ptr [rsp + 256], xmm0
	movaps xmmword ptr [rsp + 272], xmm0
	movaps xmmword ptr [rsp + 288], xmm0
	movaps xmmword ptr [rsp + 304], xmm0
	movaps xmmword ptr [rsp + 320], xmm0
	movaps xmmword ptr [rsp + 336], xmm0
	movaps xmmword ptr [rsp + 352], xmm0
	movaps xmmword ptr [rsp + 368], xmm0
	mov qword ptr [rsp + 383], 0
.Ltmp26883:
	mov dword ptr [rsp + 192], -2147483648
	mov byte ptr [rsp + 196], 0
	movabs rax, 81985529216486880
	mov qword ptr [rsp + 248], rax
	movaps xmmword ptr [rsp + 512], xmm0
	movaps xmmword ptr [rsp + 528], xmm0
.Ltmp26884:
	mov edi, 384
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp26885:
	test rax, rax
	je .LBB226_124
.Ltmp26886:
	mov r14, rax
	lea rsi, [rsp + 192]
	mov edx, 384
	mov rdi, rax
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp26887:
	or dword ptr [r15], 4
.Ltmp26888:
	#MEMBARRIER
.Ltmp26783:
	lea rdi, [rsp + 192]
	mov qword ptr [rsp + 632], rdi
.Ltmp26889:
	mov rsi, r13
	mov rdx, qword ptr [rsp + 80]
	mov rcx, r14
	mov r8, rbx
	call <masstree::leaf15::LeafNode15<S> as masstree::leaf_trait::TreeLeafNode<S>>::split_into_preallocated
.Ltmp26784:
	mov r14, qword ptr [rsp + 192]
.Ltmp26891:
	mov rax, qword ptr [rsp + 200]
	mov qword ptr [rsp + 96], rax
.Ltmp26893:
	mov rdx, qword ptr [rsp + 72]
.Ltmp26894:
	lea rax, [rdx + 960]
	mov qword ptr [rsp + 80], rax
.Ltmp26895:
	mov cl, 1
.Ltmp26896:
	xor eax, eax
	lock cmpxchg	byte ptr [rdx + 960], cl
.Ltmp26897:
	jne .LBB226_125
.Ltmp26898:
.LBB226_7:
	mov rdx, qword ptr [rsp + 72]
.Ltmp26899:
	mov rbx, qword ptr [rdx + 984]
.Ltmp26900:
	cmp rbx, qword ptr [rdx + 968]
	jne .LBB226_9
.Ltmp26901:
.Ltmp26787:
	lea rdi, [rdx + 968]
.Ltmp26902:
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp26903:
	mov rdx, qword ptr [rsp + 72]
.Ltmp26788:
.Ltmp26904:
.LBB226_9:
	mov rax, qword ptr [rdx + 976]
.Ltmp26905:
	mov qword ptr [rax + 8*rbx], r14
.Ltmp26906:
	inc rbx
.Ltmp26907:
	mov qword ptr [rdx + 984], rbx
	xor ecx, ecx
.Ltmp26908:
	mov al, 1
	lock cmpxchg	byte ptr [rdx + 960], cl
.Ltmp26909:
	je .LBB226_12
.Ltmp26793:
	mov rdi, qword ptr [rsp + 80]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26794:
	jmp .LBB226_12
.Ltmp26911:
	.p2align	4
.LBB226_11:
	mov rdi, r13
	call masstree::leaf15::LeafNode15<S>::wait_for_split
.Ltmp26913:
.LBB226_12:
	mov rcx, qword ptr [r13 + 328]
.Ltmp26914:
	test cl, 1
	jne .LBB226_11
.Ltmp26915:
.LBB226_13:
	lea rdx, [rcx + 1]
.Ltmp26916:
	mov rax, rcx
	lock cmpxchg	qword ptr [r13 + 328], rdx
.Ltmp26917:
	je .LBB226_17
.Ltmp26918:
	pause
.Ltmp26919:
	mov rcx, qword ptr [r13 + 328]
.Ltmp26920:
	test cl, 1
	jne .LBB226_11
	jmp .LBB226_13
.Ltmp26921:
.LBB226_15:
	test r12b, 4
	jne .LBB226_76
	mov eax, r12d
	and eax, 2
	lea r12d, [r12 + 4*rax]
.Ltmp26923:
	and r12d, -268435464
	jmp .LBB226_77
.Ltmp26924:
.LBB226_17:
	mov qword ptr [r14 + 336], r13
.Ltmp26925:
	mov qword ptr [r14 + 328], rcx
.Ltmp26926:
	test rcx, rcx
.Ltmp26927:
	je .LBB226_19
.Ltmp26928:
	mov qword ptr [rcx + 336], r14
.Ltmp26929:
.LBB226_19:
	#MEMBARRIER
	mov qword ptr [r13 + 328], r14
	mov esi, dword ptr [rsp + 108]
	mov byte ptr [rsp + 54], sil
	mov eax, dword ptr [rsp + 68]
	mov byte ptr [rsp + 55], al
	mov rax, qword ptr [rsp + 72]
	lea rcx, [rax + 992]
	mov qword ptr [rsp + 40], rcx
	add rax, 1000
	mov qword ptr [rsp + 112], rax
	mov eax, 1
	mov edi, 344
	mov cl, 1
	jmp .LBB226_21
.Ltmp26931:
	.p2align	4
.LBB226_20:
	test dl, 1
	mov rax, r14
	cmovne rax, r15
.Ltmp26932:
	mov rdx, qword ptr [rsp + 56]
.Ltmp26933:
	mov rcx, qword ptr [rsp + 80]
.Ltmp26934:
	mov qword ptr [rdx + rcx], rax
.Ltmp26935:
	mov eax, dword ptr [rdx]
.Ltmp26936:
	add eax, 512
	and eax, -1342177792
.Ltmp26937:
	mov ecx, dword ptr [rsp + 92]
.Ltmp26938:
	add ecx, 512
	and ecx, -1342177792
.Ltmp26939:
	cmp qword ptr [rsp + 640], r15
.Ltmp26940:
	#MEMBARRIER
	mov dword ptr [rdx], eax
	mov rax, qword ptr [rsp + 120]
.Ltmp26941:
	mov dword ptr [rax], ecx
.Ltmp26942:
	sete byte ptr [rsp + 54]
	mov eax, dword ptr [rsp + 68]
	mov byte ptr [rsp + 55], al
.Ltmp26943:
	mov rdx, qword ptr [rsp + 184]
.Ltmp26944:
	lea rax, [rdx + 1]
	xor ecx, ecx
	mov r13, r15
	cmp rdx, 63
	mov esi, dword ptr [rsp + 108]
	mov edi, 344
	ja .LBB226_136
.Ltmp26945:
.LBB226_21:
	mov qword ptr [rsp + 56], r14
.Ltmp26946:
	mov dword ptr [rsp + 92], r12d
.Ltmp26947:
	test cl, 1
	mov edx, 264
	cmovne rdx, rdi
	mov qword ptr [rsp + 80], rdx
	mov qword ptr [rsp + 120], r15
.Ltmp26948:
	not sil
	mov rdi, rax
	xor edx, edx
	mov r8d, esi
	jmp .LBB226_25
.Ltmp26949:
	.p2align	4
.LBB226_22:
	mov rax, qword ptr [rsp + 80]
.Ltmp26950:
	mov rax, qword ptr [r13 + rax]
.Ltmp26951:
	cmp rax, r15
	je .LBB226_36
	xor edx, edx
.Ltmp26953:
.LBB226_24:
	lea eax, [r12 + 512]
	mov esi, r12d
	add esi, 8
	test r12b, 4
	mov edi, -1342177792
	mov r9d, -268435464
	cmove edi, r9d
	cmovne esi, eax
	and esi, edi
	mov dword ptr [r15], esi
.Ltmp26954:
	pause
.Ltmp26955:
	mov rax, qword ptr [rsp + 184]
.Ltmp26956:
	cmp rax, 63
	lea rdi, [rax + 1]
.Ltmp26957:
	ja .LBB226_135
.Ltmp26958:
.LBB226_25:
	mov rax, qword ptr [rsp + 80]
.Ltmp26959:
	mov r15, qword ptr [r13 + rax]
.Ltmp26960:
	test r15, r15
.Ltmp26961:
	jne .LBB226_27
.Ltmp26962:
	test byte ptr [rsp + 68], 1
	jne .LBB226_78
.Ltmp26963:
.LBB226_27:
	test r15, r15
	setne al
.Ltmp26964:
	or al, r8b
	test al, 1
	je .LBB226_85
.Ltmp26965:
	test r15, r15
.Ltmp26966:
	je .LBB226_134
.Ltmp26967:
	mov qword ptr [rsp + 184], rdi
	xor esi, esi
	jmp .LBB226_31
.Ltmp26968:
	.p2align	4
.LBB226_30:
	and esi, 7
	lea esi, [2*rsi + 1]
.LBB226_31:
	mov r12d, dword ptr [r15]
	test r12b, 1
	jne .LBB226_33
	mov r14d, r12d
	or r14d, 3
	mov eax, r12d
	lock cmpxchg	dword ptr [r15], r14d
	je .LBB226_22
.LBB226_33:
	xor eax, eax
	.p2align	4
.LBB226_34:
	mov edi, eax
	pause
	cmp eax, esi
	adc eax, 0
	cmp edi, esi
	jae .LBB226_30
	cmp eax, esi
	jbe .LBB226_34
	jmp .LBB226_30
.Ltmp26974:
	.p2align	4
.LBB226_36:
	movzx eax, byte ptr [r15 + 4]
.Ltmp26976:
	movzx ebx, al
.Ltmp26977:
	test bl, bl
.Ltmp26978:
	je .LBB226_43
	xor eax, eax
	jmp .LBB226_38
.Ltmp26980:
	.p2align	4
.LBB226_40:
	mov rsi, qword ptr [r15 + 256]
.Ltmp26981:
	cmp rsi, r13
.Ltmp26982:
	je .LBB226_46
.Ltmp26983:
.LBB226_41:
	inc rax
.Ltmp26984:
	cmp rbx, rax
.Ltmp26985:
	je .LBB226_42
.Ltmp26986:
.LBB226_38:
	cmp rax, 15
	jae .LBB226_40
.Ltmp26987:
	mov rsi, qword ptr [r15 + 8*rax + 136]
.Ltmp26988:
	cmp rsi, r13
.Ltmp26989:
	jne .LBB226_41
	jmp .LBB226_46
.Ltmp26990:
.LBB226_42:
	dec rax
.Ltmp26991:
	cmp rax, 14
	jae .LBB226_44
.Ltmp26992:
.LBB226_43:
	mov rax, qword ptr [r15 + 8*rbx + 136]
.Ltmp26993:
	cmp rax, r13
.Ltmp26994:
	jne .LBB226_45
	jmp .LBB226_47
.Ltmp26995:
.LBB226_44:
	mov rax, qword ptr [r15 + 256]
.Ltmp26996:
	cmp rax, r13
.Ltmp26997:
	je .LBB226_47
.Ltmp26998:
.LBB226_45:
	inc rdx
	cmp rdx, 16
	jbe .LBB226_24
	jmp .LBB226_110
	.p2align	4
.LBB226_46:
	mov rbx, rax
.Ltmp27002:
.LBB226_47:
	movzx eax, byte ptr [r15 + 4]
.Ltmp27003:
	cmp al, 14
.Ltmp27004:
	jbe .LBB226_111
.Ltmp27005:
	or dword ptr [r15], 4
.Ltmp27006:
	#MEMBARRIER
	mov rax, qword ptr [rsp + 72]
.Ltmp27007:
	mov rdx, qword ptr [rax + 1024]
.Ltmp27008:
	cmp rdx, r15
	sete cl
	mov dword ptr [rsp + 108], ecx
.Ltmp27009:
	mov qword ptr [rsp + 640], rdx
.Ltmp27010:
	je .LBB226_51
.Ltmp27011:
	mov rax, qword ptr [r15 + 264]
.Ltmp27012:
	test rax, rax
.Ltmp27013:
	je .LBB226_52
.LBB226_51:
	mov dword ptr [rsp + 68], 0
	jmp .LBB226_53
.Ltmp27015:
.LBB226_52:
	mov eax, dword ptr [r15]
.Ltmp27016:
	shr eax, 30
.Ltmp27017:
	and al, 1
	mov dword ptr [rsp + 68], eax
.Ltmp27018:
.LBB226_53:
	or r12d, 7
.Ltmp27019:
	mov eax, dword ptr [r15 + 8]
.Ltmp27020:
	mov dword ptr [rsp + 148], eax
.Ltmp27021:
	mov r13d, dword ptr [r15]
.Ltmp27022:
	lea rax, [rsp + 196]
	xorps xmm0, xmm0
.Ltmp27023:
	movups xmmword ptr [rax + 96], xmm0
	movups xmmword ptr [rax + 80], xmm0
	movups xmmword ptr [rax + 64], xmm0
	movups xmmword ptr [rax + 48], xmm0
	movups xmmword ptr [rax + 32], xmm0
	movups xmmword ptr [rax + 16], xmm0
	movups xmmword ptr [rax], xmm0
	mov qword ptr [rax + 112], 0
.Ltmp27024:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp27025:
	test rax, rax
	je .LBB226_123
.Ltmp27026:
	mov r14, rax
	and r13d, -2147483648
	or r13d, 5
.Ltmp27027:
	mov dword ptr [rax], r13d
	mov byte ptr [rax + 4], 0
	mov eax, dword ptr [rsp + 148]
	mov dword ptr [r14 + 8], eax
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [r14 + 12], xmm0
	movups xmmword ptr [r14 + 28], xmm1
	movups xmmword ptr [r14 + 44], xmm2
	movups xmmword ptr [r14 + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [r14 + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [r14 + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [r14 + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [r14 + 120], xmm0
	xorps xmm0, xmm0
	movups xmmword ptr [r14 + 136], xmm0
	movups xmmword ptr [r14 + 152], xmm0
	movups xmmword ptr [r14 + 168], xmm0
	movups xmmword ptr [r14 + 184], xmm0
	movups xmmword ptr [r14 + 200], xmm0
	movups xmmword ptr [r14 + 216], xmm0
	movups xmmword ptr [r14 + 232], xmm0
	movups xmmword ptr [r14 + 248], xmm0
	mov qword ptr [r14 + 264], 0
.Ltmp27028:
	xor eax, eax
	mov rcx, qword ptr [rsp + 40]
	mov dl, 1
	lock cmpxchg	byte ptr [rcx], dl
.Ltmp27029:
	jne .LBB226_74
.Ltmp27030:
.LBB226_55:
	mov rax, qword ptr [rsp + 72]
.Ltmp27031:
	mov r13, qword ptr [rax + 1016]
.Ltmp27032:
	cmp r13, qword ptr [rax + 1000]
	jne .LBB226_57
.Ltmp26800:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp26801:
.LBB226_57:
	mov rcx, qword ptr [rsp + 72]
.Ltmp27035:
	mov rax, qword ptr [rcx + 1008]
.Ltmp27036:
	mov qword ptr [rax + 8*r13], r14
.Ltmp27037:
	inc r13
.Ltmp27038:
	mov qword ptr [rcx + 1016], r13
.Ltmp27039:
	mov al, 1
	xor edx, edx
	lock cmpxchg	byte ptr [rcx + 992], dl
.Ltmp27040:
	jne .LBB226_75
.Ltmp27041:
.LBB226_58:
.Ltmp26808:
	mov rdi, r15
	mov rsi, r14
	mov rdx, r14
	mov rcx, rbx
	mov r8, qword ptr [rsp + 96]
	mov r9, qword ptr [rsp + 56]
	call masstree::internode::InternodeNode<S,_>::split_into
	mov qword ptr [rsp + 96], rax
.Ltmp27042:
	lea rax, [r14 + 136]
.Ltmp27043:
	cmp dword ptr [r15 + 8], 0
.Ltmp27044:
	movzx ecx, byte ptr [r14 + 4]
.Ltmp27045:
	movzx ecx, cl
.Ltmp27046:
	je .LBB226_67
	xor edi, edi
.Ltmp27048:
	xor esi, esi
.Ltmp27049:
	.p2align	4
.LBB226_61:
	cmp rdi, rcx
.Ltmp27051:
	adc rsi, 0
.Ltmp27052:
	cmp rdi, 15
	jae .LBB226_63
.Ltmp27053:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp27054:
	test r8, r8
.Ltmp27055:
	jne .LBB226_64
	jmp .LBB226_65
.Ltmp27056:
	.p2align	4
.LBB226_63:
	mov r8, qword ptr [r14 + 256]
.Ltmp27058:
	test r8, r8
.Ltmp27059:
	je .LBB226_65
.Ltmp27060:
.LBB226_64:
	mov qword ptr [r8 + 264], r14
.Ltmp27061:
.LBB226_65:
	cmp rdi, rcx
.Ltmp27062:
	jae .LBB226_20
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB226_61
	jmp .LBB226_20
.Ltmp27064:
	.p2align	4
.LBB226_67:
	xor edi, edi
.Ltmp27066:
	xor esi, esi
.Ltmp27067:
	.p2align	4
.LBB226_68:
	cmp rdi, rcx
.Ltmp27069:
	adc rsi, 0
.Ltmp27070:
	cmp rdi, 15
	jae .LBB226_70
.Ltmp27071:
	mov r8, qword ptr [rax + 8*rdi]
.Ltmp27072:
	test r8, r8
.Ltmp27073:
	jne .LBB226_71
	jmp .LBB226_72
.Ltmp27074:
	.p2align	4
.LBB226_70:
	mov r8, qword ptr [r14 + 256]
.Ltmp27076:
	test r8, r8
.Ltmp27077:
	je .LBB226_72
.Ltmp27078:
.LBB226_71:
	mov qword ptr [r8 + 344], r14
.Ltmp27079:
.LBB226_72:
	cmp rdi, rcx
.Ltmp27080:
	jae .LBB226_20
	mov rdi, rsi
	cmp rsi, rcx
	jbe .LBB226_68
	jmp .LBB226_20
.Ltmp27082:
.LBB226_74:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27083:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB226_55
.Ltmp27084:
.LBB226_75:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27085:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26807:
	jmp .LBB226_58
.Ltmp27086:
.LBB226_76:
	add r12d, 512
.Ltmp27087:
	and r12d, -1342177792
.LBB226_77:
	mov dword ptr [r15], r12d
	mov al, 4
.Ltmp27089:
	jmp .LBB226_109
.Ltmp27090:
.LBB226_78:
	test cl, 1
	je .LBB226_93
.Ltmp27091:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp27092:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp27093:
	test rax, rax
	mov r15, qword ptr [rsp + 72]
.Ltmp27094:
	je .LBB226_126
.Ltmp27095:
	mov rbx, rax
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp27096:
	xorps xmm0, xmm0
.Ltmp27097:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp27098:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 96]
.Ltmp27100:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 56]
.Ltmp27102:
	mov qword ptr [rbx + 144], rax
.Ltmp27103:
	mov byte ptr [rbx + 4], 1
.Ltmp27104:
	lock or	dword ptr [rbx], 1073741824
	mov cl, 1
.Ltmp27105:
	xor eax, eax
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27106:
	jne .LBB226_127
.Ltmp27107:
.LBB226_81:
	mov r14, qword ptr [r15 + 1016]
.Ltmp27108:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB226_83
.Ltmp26848:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp26849:
.Ltmp27110:
.LBB226_83:
	mov rax, qword ptr [r15 + 1008]
.Ltmp27111:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp27112:
	inc r14
.Ltmp27113:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp27114:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp27115:
	jne .LBB226_129
.Ltmp27116:
.LBB226_84:
	#MEMBARRIER
	mov qword ptr [r13 + 344], rbx
	mov rcx, qword ptr [rsp + 56]
.Ltmp27118:
	mov qword ptr [rcx + 344], rbx
.Ltmp27119:
	lock and	dword ptr [r13], -1073741825
	mov r13, rcx
.Ltmp27121:
	jmp .LBB226_106
.Ltmp27122:
.LBB226_85:
	test cl, 1
	je .LBB226_99
.Ltmp27123:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp27124:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp27125:
	test rax, rax
	mov r15, qword ptr [rsp + 72]
.Ltmp27126:
	je .LBB226_126
.Ltmp27127:
	mov rbx, rax
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp27128:
	xorps xmm0, xmm0
.Ltmp27129:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp27130:
	lock or	dword ptr [rax], 1073741824
.Ltmp27131:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 96]
.Ltmp27133:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 56]
.Ltmp27135:
	mov qword ptr [rbx + 144], rax
.Ltmp27136:
	mov byte ptr [rbx + 4], 1
	mov cl, 1
.Ltmp27137:
	xor eax, eax
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27138:
	jne .LBB226_130
.Ltmp27139:
.LBB226_88:
	mov r14, qword ptr [r15 + 1016]
.Ltmp27140:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB226_90
.Ltmp26826:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp26827:
.Ltmp27142:
.LBB226_90:
	mov rax, qword ptr [r15 + 1008]
.Ltmp27143:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp27144:
	inc r14
.Ltmp27145:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp27146:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp27147:
	jne .LBB226_132
.Ltmp27148:
.LBB226_91:
	mov qword ptr [rsp + 128], r13
.Ltmp27149:
	mov rax, r13
	lock cmpxchg	qword ptr [r15 + 1024], rbx
.Ltmp27150:
	jne .LBB226_141
.Ltmp27151:
	#MEMBARRIER
	mov qword ptr [r13 + 344], rbx
	mov rcx, qword ptr [rsp + 56]
.Ltmp27153:
	mov qword ptr [rcx + 344], rbx
.Ltmp27154:
	jmp .LBB226_106
.Ltmp27155:
.LBB226_93:
	mov r14d, dword ptr [r13 + 8]
.Ltmp27156:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp27157:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp27158:
	test rax, rax
	mov r15, qword ptr [rsp + 72]
.Ltmp27159:
	je .LBB226_126
.Ltmp27160:
	mov rbx, rax
	inc r14d
.Ltmp27161:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r14d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp27162:
	xorps xmm0, xmm0
.Ltmp27163:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp27164:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 96]
.Ltmp27166:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 56]
.Ltmp27168:
	mov qword ptr [rbx + 144], rax
.Ltmp27169:
	mov byte ptr [rbx + 4], 1
.Ltmp27170:
	lock or	dword ptr [rbx], 1073741824
	mov cl, 1
.Ltmp27171:
	xor eax, eax
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27172:
	jne .LBB226_128
.Ltmp27173:
.LBB226_95:
	mov r14, qword ptr [r15 + 1016]
.Ltmp27174:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB226_97
.Ltmp26838:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp26839:
.Ltmp27176:
.LBB226_97:
	mov rax, qword ptr [r15 + 1008]
.Ltmp27177:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp27178:
	inc r14
.Ltmp27179:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp27180:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp27181:
	je .LBB226_105
.Ltmp26844:
	mov rdi, qword ptr [rsp + 40]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26845:
	jmp .LBB226_105
.Ltmp27183:
.LBB226_99:
	mov r14d, dword ptr [r13 + 8]
.Ltmp27184:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp27185:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp27186:
	test rax, rax
	mov r15, qword ptr [rsp + 72]
.Ltmp27187:
	je .LBB226_126
.Ltmp27188:
	mov rbx, rax
	inc r14d
.Ltmp27189:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r14d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp27190:
	xorps xmm0, xmm0
.Ltmp27191:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp27192:
	lock or	dword ptr [rax], 1073741824
.Ltmp27193:
	mov qword ptr [rax + 136], r13
	mov rax, qword ptr [rsp + 96]
.Ltmp27195:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 56]
.Ltmp27197:
	mov qword ptr [rbx + 144], rax
.Ltmp27198:
	mov byte ptr [rbx + 4], 1
	mov cl, 1
.Ltmp27199:
	xor eax, eax
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27200:
	jne .LBB226_131
.Ltmp27201:
.LBB226_101:
	mov r14, qword ptr [r15 + 1016]
.Ltmp27202:
	cmp r14, qword ptr [r15 + 1000]
	jne .LBB226_103
.Ltmp26816:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp26817:
.Ltmp27204:
.LBB226_103:
	mov rax, qword ptr [r15 + 1008]
.Ltmp27205:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp27206:
	inc r14
.Ltmp27207:
	mov qword ptr [r15 + 1016], r14
	xor ecx, ecx
.Ltmp27208:
	mov al, 1
	lock cmpxchg	byte ptr [r15 + 992], cl
.Ltmp27209:
	jne .LBB226_133
.Ltmp27210:
.LBB226_104:
	mov qword ptr [rsp + 128], r13
.Ltmp27211:
	mov rax, r13
	lock cmpxchg	qword ptr [r15 + 1024], rbx
.Ltmp27212:
	jne .LBB226_142
.Ltmp27213:
.LBB226_105:
	#MEMBARRIER
	mov qword ptr [r13 + 264], rbx
	mov rcx, qword ptr [rsp + 56]
	mov qword ptr [rcx + 264], rbx
.Ltmp27215:
.LBB226_106:
	lock and	dword ptr [r13], -1073741825
.Ltmp27216:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
.Ltmp27217:
.LBB226_107:
	mov al, 6
.LBB226_108:
	mov ecx, dword ptr [rsp + 92]
.Ltmp27219:
	add ecx, 512
	and ecx, -1342177792
	mov rdx, qword ptr [rsp + 120]
	mov dword ptr [rdx], ecx
.Ltmp27220:
.LBB226_109:
	lea rsp, [rbp - 40]
.Ltmp27221:
	pop rbx
	pop r12
	pop r13
	pop r14
	pop r15
	pop rbp
	.cfi_def_cfa rsp, 8
	ret
.Ltmp27222:
.LBB226_110:
	.cfi_def_cfa rbp, 16
	mov rdi, qword ptr [rsp + 56]
.Ltmp27223:
	mov eax, dword ptr [rdi]
	add eax, 512
	and eax, -1342177792
.Ltmp27224:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
.Ltmp27225:
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp27226:
	#MEMBARRIER
	mov dword ptr [rdi], eax
.Ltmp27227:
	and ecx, esi
	mov dword ptr [r15], ecx
	mov al, 4
	jmp .LBB226_108
.Ltmp27229:
.LBB226_111:
	movzx eax, byte ptr [r15 + 4]
.Ltmp27230:
	movzx eax, al
.Ltmp27231:
	cmp rbx, rax
.Ltmp27232:
	jae .LBB226_118
.Ltmp27233:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.206]
	mov rcx, rax
	jmp .LBB226_114
.Ltmp27234:
	.p2align	4
.LBB226_113:
	mov qword ptr [r15 + 8*rsi + 152], rdi
	mov rcx, rsi
.Ltmp27236:
	cmp rbx, rsi
.Ltmp27237:
	jae .LBB226_118
.Ltmp27238:
.LBB226_114:
	cmp rcx, 16
	jae .LBB226_138
.Ltmp27239:
	lea rsi, [rcx - 1]
.Ltmp27240:
	mov rdi, qword ptr [r15 + 8*rsi + 16]
.Ltmp27241:
	cmp rcx, 15
	je .LBB226_139
.Ltmp27242:
	mov qword ptr [r15 + 8*rsi + 24], rdi
.Ltmp27243:
	mov rdi, qword ptr [r15 + 8*rsi + 144]
.Ltmp27244:
	cmp rcx, 14
	jb .LBB226_113
.Ltmp27245:
	mov qword ptr [r15 + 256], rdi
	mov rcx, rsi
.Ltmp27247:
	cmp rbx, rsi
.Ltmp27248:
	jb .LBB226_114
.Ltmp27249:
.LBB226_118:
	cmp rbx, 14
	ja .LBB226_145
	mov rcx, qword ptr [rsp + 96]
.Ltmp27251:
	mov qword ptr [r15 + 8*rbx + 16], rcx
.Ltmp27252:
	jne .LBB226_121
	mov rdi, qword ptr [rsp + 56]
.Ltmp27254:
	mov qword ptr [r15 + 256], rdi
.Ltmp27255:
	jmp .LBB226_122
.LBB226_121:
	mov rdi, qword ptr [rsp + 56]
.Ltmp27257:
	mov qword ptr [r15 + 8*rbx + 144], rdi
.Ltmp27258:
.LBB226_122:
	#MEMBARRIER
	inc al
.Ltmp27259:
	mov byte ptr [r15 + 4], al
	mov rax, qword ptr [rsp + 80]
.Ltmp27261:
	mov qword ptr [rdi + rax], r15
.Ltmp27262:
	mov eax, dword ptr [rdi]
	add eax, 512
	and eax, -1342177792
.Ltmp27263:
	mov ecx, r12d
	add ecx, 8
	test r12b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r12 + 512]
	cmovne ecx, edx
.Ltmp27264:
	#MEMBARRIER
	mov dword ptr [rdi], eax
.Ltmp27265:
	and ecx, esi
	mov dword ptr [r15], ecx
	jmp .LBB226_107
.Ltmp27266:
.LBB226_123:
.Ltmp26811:
	mov edi, 64
	mov esi, 320
	mov r14d, r12d
	call alloc::alloc::handle_alloc_error
.Ltmp26812:
	jmp .LBB226_144
.Ltmp27267:
.LBB226_124:
.Ltmp26859:
	mov edi, 64
	mov esi, 384
	call alloc::alloc::handle_alloc_error
.Ltmp26860:
	jmp .LBB226_144
.Ltmp27268:
.LBB226_125:
.Ltmp26785:
	mov rdi, qword ptr [rsp + 80]
.Ltmp27269:
	call parking_lot::raw_mutex::RawMutex::lock_slow
.Ltmp26786:
	jmp .LBB226_7
.Ltmp27270:
.LBB226_126:
.Ltmp26856:
	mov edi, 64
	mov esi, 320
	call alloc::alloc::handle_alloc_error
	jmp .LBB226_144
.Ltmp27271:
.LBB226_127:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27272:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB226_81
.Ltmp27273:
.LBB226_128:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27274:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB226_95
.Ltmp27275:
.LBB226_129:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27276:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
	jmp .LBB226_84
.Ltmp27277:
.LBB226_130:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27278:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB226_88
.Ltmp27279:
.LBB226_131:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27280:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB226_101
.Ltmp27281:
.LBB226_132:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27282:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
	jmp .LBB226_91
.Ltmp27283:
.LBB226_133:
	mov rdi, qword ptr [rsp + 40]
.Ltmp27284:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26823:
	jmp .LBB226_104
.Ltmp27285:
.LBB226_134:
	mov rcx, qword ptr [rsp + 56]
.Ltmp27286:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
.Ltmp27287:
	mov eax, dword ptr [rsp + 92]
.Ltmp27288:
	add eax, 512
	and eax, -1342177792
	mov rcx, qword ptr [rsp + 120]
	mov dword ptr [rcx], eax
	lea rax, [rsp + 54]
.Ltmp27290:
	mov qword ptr [rsp + 152], rax
	lea rcx, [rip + <bool as core::fmt::Display>::fmt]
	mov qword ptr [rsp + 160], rcx
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.166]
	mov eax, 2
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.164]
.Ltmp27291:
	lea r8, [rsp + 152]
	mov edi, 24
	lea r10, [rsp + 55]
	lea r9, [rsp + 168]
	jmp .LBB226_137
.Ltmp27292:
.LBB226_135:
	mov r15, qword ptr [rsp + 120]
	mov eax, dword ptr [rsp + 92]
	mov r12d, eax
	mov r14, qword ptr [rsp + 56]
.Ltmp27293:
.LBB226_136:
	mov eax, dword ptr [r14]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [r14], eax
.Ltmp27294:
	add r12d, 512
.Ltmp27295:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.171]
	mov eax, 1
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.170]
	lea rcx, [rip + core::fmt::num::imp::<impl core::fmt::Display for u64>::fmt]
	lea r9, [rsp + 152]
	mov edi, 8
	lea r10, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.167]
	mov r8, r9
.Ltmp27296:
.LBB226_137:
	mov qword ptr [r9], r10
	mov qword ptr [r8 + rdi], rcx
	mov rdi, qword ptr [rsp + 632]
	mov qword ptr [rdi], rdx
	mov qword ptr [rdi + 8], 2
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], r8
	mov qword ptr [rdi + 24], rax
	call core::panicking::panic_fmt
.Ltmp27297:
.LBB226_138:
	dec rcx
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.205]
	mov rbx, rcx
.Ltmp27298:
	jmp .LBB226_140
.Ltmp27299:
.LBB226_139:
	mov ebx, 15
.Ltmp27300:
.LBB226_140:
.Ltmp26796:
	mov esi, 15
	mov rdi, rbx
	call core::panicking::panic_bounds_check
.Ltmp27301:
.Ltmp26797:
	jmp .LBB226_144
.Ltmp27302:
.LBB226_141:
	mov qword ptr [rsp + 136], rax
	lea rax, [rsp + 128]
.Ltmp27304:
	mov qword ptr [rsp + 152], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 160], rax
	lea rcx, [rsp + 136]
	mov qword ptr [rsp + 168], rcx
	mov qword ptr [rsp + 176], rax
.Ltmp27305:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.175]
.Ltmp27306:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.177]
	jmp .LBB226_143
.Ltmp27307:
.LBB226_142:
	mov qword ptr [rsp + 136], rax
	lea rax, [rsp + 128]
.Ltmp27309:
	mov qword ptr [rsp + 152], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 160], rax
	lea rcx, [rsp + 136]
	mov qword ptr [rsp + 168], rcx
	mov qword ptr [rsp + 176], rax
.Ltmp27310:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.180]
.Ltmp27311:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.181]
.Ltmp27312:
.LBB226_143:
	lea rax, [rsp + 152]
	lea rdi, [rsp + 192]
	mov qword ptr [rdi + 8], 3
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], rax
	mov qword ptr [rdi + 24], 2
.Ltmp26834:
	call core::panicking::panic_fmt
.Ltmp27313:
.Ltmp26835:
.LBB226_144:
	ud2
.Ltmp27314:
.LBB226_145:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.204]
.Ltmp27315:
	jmp .LBB226_140
.Ltmp27316:
.Ltmp26818:
	mov r13, rax
.Ltmp27317:
	xor ecx, ecx
.Ltmp27318:
	mov al, 1
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27319:
	je .LBB226_176
.Ltmp26819:
	mov rdi, qword ptr [rsp + 40]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26820:
	jmp .LBB226_176
.Ltmp27321:
.Ltmp26821:
	call core::panicking::panic_in_cleanup
.Ltmp27322:
.Ltmp26828:
	mov r13, rax
.Ltmp27323:
	xor ecx, ecx
.Ltmp27324:
	mov al, 1
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27325:
	je .LBB226_176
.Ltmp26829:
	mov rdi, qword ptr [rsp + 40]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26830:
	jmp .LBB226_176
.Ltmp27327:
.Ltmp26831:
	call core::panicking::panic_in_cleanup
.Ltmp27328:
.Ltmp26840:
	mov r13, rax
.Ltmp27329:
	xor ecx, ecx
.Ltmp27330:
	mov al, 1
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27331:
	je .LBB226_176
.Ltmp26841:
	mov rdi, qword ptr [rsp + 40]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26842:
	jmp .LBB226_176
.Ltmp27333:
.Ltmp26843:
	call core::panicking::panic_in_cleanup
.Ltmp27334:
.Ltmp26850:
	mov r13, rax
.Ltmp27335:
	xor ecx, ecx
.Ltmp27336:
	mov al, 1
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27337:
	je .LBB226_176
.Ltmp26851:
	mov rdi, qword ptr [rsp + 40]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26852:
	jmp .LBB226_176
.Ltmp27339:
.Ltmp26853:
	call core::panicking::panic_in_cleanup
.Ltmp27340:
.Ltmp26789:
	mov r13, rax
.Ltmp27341:
	xor ecx, ecx
.Ltmp27342:
	mov al, 1
	mov rdx, qword ptr [rsp + 80]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27343:
	je .LBB226_171
.Ltmp26790:
	mov rdi, qword ptr [rsp + 80]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26791:
	jmp .LBB226_171
.Ltmp27345:
.Ltmp26792:
	call core::panicking::panic_in_cleanup
.Ltmp27346:
.Ltmp26802:
	mov r13, rax
	xor ecx, ecx
.Ltmp27347:
	mov al, 1
	mov rdx, qword ptr [rsp + 40]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp27348:
	je .LBB226_173
.Ltmp26803:
	mov rdi, qword ptr [rsp + 40]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp26804:
	jmp .LBB226_173
.Ltmp27350:
.Ltmp26805:
	call core::panicking::panic_in_cleanup
.Ltmp27351:
.Ltmp26795:
	mov r13, rax
.Ltmp27352:
	jmp .LBB226_171
.Ltmp27353:
.Ltmp26810:
	mov r13, rax
	jmp .LBB226_173
.Ltmp27354:
.Ltmp26782:
	mov r13, rax
.Ltmp27355:
	jmp .LBB226_169
.Ltmp27356:
.Ltmp26858:
	mov r13, rax
	jmp .LBB226_176
.Ltmp27357:
.Ltmp26861:
	mov r13, rax
.Ltmp27358:
	lea rdi, [rsp + 192]
.Ltmp27359:
	call core::ptr::drop_in_place<masstree::leaf15::LeafNode15<masstree::value::LeafValue<u64>>>
.Ltmp27360:
.LBB226_169:
	test r12b, 4
	jne .LBB226_171
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
	mov rdi, r13
	call _Unwind_Resume@PLT
.Ltmp27362:
.LBB226_171:
	add r12d, 512
.Ltmp27363:
	and r12d, -1342177792
	mov dword ptr [r15], r12d
	mov rdi, r13
	call _Unwind_Resume@PLT
.Ltmp27364:
.Ltmp26813:
	mov r13, rax
	mov r12d, r14d
.Ltmp27365:
.LBB226_173:
	test r12b, 4
	jne .LBB226_175
	mov eax, r12d
	and eax, 2
	lea eax, [r12 + 4*rax]
	and eax, -268435464
	mov dword ptr [r15], eax
.Ltmp27367:
	jmp .LBB226_176
.Ltmp27368:
.LBB226_175:
	add r12d, 512
	and r12d, -1342177792
	mov dword ptr [r15], r12d
.Ltmp27369:
.LBB226_176:
	mov eax, dword ptr [rsp + 92]
.Ltmp27370:
	add eax, 512
	and eax, -1342177792
	mov rcx, qword ptr [rsp + 120]
	mov dword ptr [rcx], eax
	mov rdi, r13
	call _Unwind_Resume@PLT
.Ltmp27371:
.Lfunc_end226:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic, .Lfunc_end226-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic","a",@progbits
	.p2align	2, 0x0
GCC_except_table226:
masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic:
.Lfunc_begin234:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception103
	push rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov rbp, rsp
	.cfi_def_cfa_register rbp
	push r15
	push r14
	push r13
	push r12
	push rbx
	and rsp, -64
	sub rsp, 832
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	mov qword ptr [rsp + 64], r9
.Ltmp28651:
	mov r14, r8
	mov r15d, ecx
	mov rbx, rdx
	mov r13, rdi
	mov qword ptr [rsp + 72], rsi
.Ltmp28653:
	lea r12, [rsi + 64]
.Ltmp28654:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp28655:
.Ltmp28555:
	mov rdi, r12
.Ltmp28656:
	call rax
.Ltmp28657:
.Ltmp28556:
	mov esi, eax
	and esi, 31
.Ltmp28658:
	cmp rsi, 2
	jb .LBB234_14
	mov r10d, esi
	shr r10d
.Ltmp28660:
	.p2align	4
.LBB234_3:
	lea r8, [r10 - 1]
.Ltmp28662:
	lea ecx, [r8 + 4*r8]
	add cl, 5
	mov r9, rax
	shrd r9, rdx, cl
	mov rdi, rdx
	shr rdi, cl
	test cl, 64
	cmove rdi, r9
.Ltmp28663:
	mov r11, r10
.Ltmp28664:
	lea ecx, [r10 + 4*r10]
	add cl, 5
	mov r10, rax
.Ltmp28665:
	shrd r10, rdx, cl
.Ltmp28666:
	and edi, 31
.Ltmp28667:
	mov r9, rdx
	shr r9, cl
	test cl, 64
	cmove r9, r10
.Ltmp28668:
	cmp rdi, 23
	ja .LBB234_222
.Ltmp28669:
	and r9d, 31
.Ltmp28670:
	mov r10, qword ptr [rsp + 72]
.Ltmp28671:
	mov rcx, qword ptr [r10 + 8*rdi + 128]
.Ltmp28672:
	cmp r9d, 23
	ja .LBB234_221
.Ltmp28673:
	mov rdi, qword ptr [r10 + 8*r9 + 128]
.Ltmp28674:
	cmp rcx, rdi
	jne .LBB234_11
.Ltmp28675:
	cmp r14, rcx
	seta cl
.Ltmp28676:
	sbb cl, 0
.Ltmp28677:
	je .LBB234_9
	movzx ecx, cl
	cmp ecx, 255
	mov r10, r11
	jne .LBB234_13
.Ltmp28679:
	test r8, r8
	jne .LBB234_10
	jmp .LBB234_12
.Ltmp28680:
	.p2align	4
.LBB234_9:
	mov r8, r11
.Ltmp28682:
	inc r8
.Ltmp28683:
	test r8, r8
	je .LBB234_12
.Ltmp28684:
.LBB234_10:
	mov r10, r8
	cmp r8, rsi
	jb .LBB234_3
	jmp .LBB234_13
.Ltmp28685:
.LBB234_11:
	mov r10, r11
	jmp .LBB234_13
.Ltmp28686:
.LBB234_12:
	mov r10, r8
.Ltmp28687:
.LBB234_13:
	test r10, r10
	sete cl
	cmp r10, rsi
	setae sil
.Ltmp28688:
	or sil, cl
	je .LBB234_18
.Ltmp28689:
.LBB234_14:
	test r15b, 4
	jne .LBB234_16
	mov eax, r15d
	and eax, 2
	lea r15d, [r15 + 4*rax]
.Ltmp28691:
	and r15d, -268435464
	jmp .LBB234_17
.Ltmp28692:
.LBB234_16:
	add r15d, 512
.Ltmp28693:
	and r15d, -1342177792
.LBB234_17:
	mov dword ptr [rbx], r15d
	mov al, 4
.Ltmp28695:
	jmp .LBB234_193
.Ltmp28696:
.LBB234_18:
	mov qword ptr [rsp + 24], r10
.Ltmp28697:
	lea ecx, [r10 + 4*r10]
	add cl, 5
	shrd rax, rdx, cl
.Ltmp28698:
	shr rdx, cl
.Ltmp28699:
	test cl, 64
	cmove rdx, rax
	mov edi, edx
	and edi, 31
.Ltmp28700:
	cmp rdi, 24
	jae .LBB234_222
	mov qword ptr [rsp + 40], r12
	mov r14, qword ptr [rsp + 72]
.Ltmp28702:
	mov rax, qword ptr [r14 + 8*rdi + 128]
.Ltmp28703:
	mov ecx, dword ptr [r14]
	mov dword ptr [rsp + 36], 0
.Ltmp28704:
	mov rax, qword ptr [r14 + 560]
.Ltmp28705:
	mov dword ptr [rsp + 92], 0
.Ltmp28706:
	test ecx, 1073741824
.Ltmp28707:
	je .LBB234_21
.Ltmp28708:
	test rax, rax
	sete al
.Ltmp28709:
	mov rcx, qword ptr [r13 + 1024]
.Ltmp28710:
	cmp rcx, r14
	sete cl
.Ltmp28711:
	mov dword ptr [rsp + 92], ecx
.Ltmp28712:
	setne cl
.Ltmp28713:
	and cl, al
	mov dword ptr [rsp + 36], ecx
.Ltmp28714:
.LBB234_21:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 229], xmm0
	movups xmmword ptr [rsp + 213], xmm0
	movups xmmword ptr [rsp + 197], xmm0
	mov qword ptr [rsp + 244], 0
	lea rdi, [rsp + 272]
	mov edx, 456
	xor esi, esi
	call qword ptr [rip + memset@GOTPCREL]
	mov dword ptr [rsp + 192], -2147483648
	mov byte ptr [rsp + 196], 0
	movabs rax, 2399420405009046
	mov qword ptr [rsp + 264], rax
	movabs rax, -4127003096615560480
	mov qword ptr [rsp + 256], rax
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 728], xmm0
	movups xmmword ptr [rsp + 744], xmm0
.Ltmp28715:
	mov edi, 576
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp28716:
	test rax, rax
	je .LBB234_209
.Ltmp28717:
	mov r12, rax
	lea rsi, [rsp + 192]
	mov edx, 576
	mov rdi, rax
	mov qword ptr [rsp + 184], rsi
	call qword ptr [rip + memcpy@GOTPCREL]
.Ltmp28718:
	or dword ptr [rbx], 4
.Ltmp28719:
	#MEMBARRIER
	mov eax, dword ptr [r14]
	and eax, -2147483648
	or eax, 5
.Ltmp28720:
	mov dword ptr [r12], eax
.Ltmp28721:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp28722:
.Ltmp28560:
	mov rdi, qword ptr [rsp + 40]
.Ltmp28723:
	call rax
	mov qword ptr [rsp + 80], rdx
.Ltmp28724:
.Ltmp28561:
	mov qword ptr [rsp + 8], rax
.Ltmp28725:
	and eax, 31
	xor edi, edi
.Ltmp28726:
	mov rcx, rax
	mov esi, 0
	sub rcx, qword ptr [rsp + 24]
.Ltmp28728:
	mov qword ptr [rsp + 48], rcx
	mov qword ptr [rsp + 104], r13
.Ltmp28729:
	mov qword ptr [rsp + 56], r12
.Ltmp28730:
	jne .LBB234_26
.Ltmp28731:
.LBB234_24:
	mov r10, rax
	mov rax, qword ptr [rsp + 24]
	mov ecx, eax
	sub ecx, dword ptr [rsp + 8]
	add rax, 23
	test cl, 1
	jne .LBB234_39
	cmp rax, r10
	jne .LBB234_40
	jmp .LBB234_49
.Ltmp28733:
.LBB234_26:
	mov qword ptr [rsp], rax
	mov rax, qword ptr [rsp + 24]
.Ltmp28734:
	lea r13, [rax + 4*rax]
	add r13, 5
	xor r12d, r12d
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.188]
	mov qword ptr [rsp + 120], rax
	jmp .LBB234_29
.Ltmp28735:
	.p2align	4
.LBB234_27:
.Ltmp28566:
	mov rsi, r14
	mov rdx, qword ptr [rsp + 64]
	call masstree::leaf24::LeafNode24<S>::clear_ksuf
.Ltmp28737:
.LBB234_28:
	inc r12
.Ltmp28738:
	add r13, 5
	cmp qword ptr [rsp + 48], r12
.Ltmp28739:
	je .LBB234_37
.Ltmp28740:
.LBB234_29:
	mov rax, qword ptr [rsp + 8]
.Ltmp28741:
	mov ecx, r13d
	mov r14, qword ptr [rsp + 80]
	shrd rax, r14, cl
	shr r14, cl
	test r13b, 64
	cmove r14, rax
	and r14d, 31
.Ltmp28742:
	cmp r14, 24
	jae .LBB234_223
	mov rdi, qword ptr [rsp + 72]
.Ltmp28744:
	mov rcx, qword ptr [rdi + 8*r14 + 128]
.Ltmp28745:
	movzx eax, byte ptr [rdi + r14 + 320]
.Ltmp28746:
	cmp r12, 24
	je .LBB234_226
	mov rdx, qword ptr [rsp + 56]
.Ltmp28748:
	mov qword ptr [rdx + 8*r12 + 128], rcx
.Ltmp28749:
	mov byte ptr [rdx + r12 + 320], al
.Ltmp28750:
	xor ecx, ecx
.Ltmp28751:
	xchg qword ptr [rdi + 8*r14 + 344], rcx
.Ltmp28752:
	mov qword ptr [rdx + 8*r12 + 344], rcx
.Ltmp28753:
	cmp al, 64
	jne .LBB234_28
.Ltmp28754:
	movzx eax, byte ptr [rdi + r14 + 320]
.Ltmp28755:
	cmp al, 64
.Ltmp28756:
	jne .LBB234_27
.Ltmp28757:
	mov r8, qword ptr [rdi + 536]
.Ltmp28758:
	test r8, r8
.Ltmp28759:
	je .LBB234_27
.Ltmp28760:
	mov edx, dword ptr [r8 + 8*r14 + 24]
	mov eax, 4294967295
	cmp rdx, rax
	je .LBB234_27
	movzx ecx, word ptr [r8 + 8*r14 + 28]
.Ltmp28763:
	lea rsi, [rcx + rdx]
.Ltmp28764:
	mov rax, qword ptr [r8 + 16]
.Ltmp28765:
	cmp rsi, rax
.Ltmp28766:
	ja .LBB234_207
.Ltmp28767:
	add rdx, qword ptr [r8 + 8]
.Ltmp28768:
	mov rdi, qword ptr [rsp + 56]
.Ltmp28769:
	mov rsi, r12
.Ltmp28770:
	mov r8, qword ptr [rsp + 64]
.Ltmp28771:
	call masstree::leaf24::LeafNode24<S>::assign_ksuf
.Ltmp28772:
.Ltmp28563:
	mov rdi, qword ptr [rsp + 72]
	jmp .LBB234_27
.Ltmp28773:
.LBB234_37:
	cmp qword ptr [rsp + 48], 24
	jne .LBB234_42
	movabs rdx, 1708387328366441304
	movabs r14, -5393897070460337128
	mov r13, qword ptr [rsp + 104]
	mov r12, qword ptr [rsp + 56]
	jmp .LBB234_49
.LBB234_39:
	mov r8, qword ptr [rsp + 48]
	lea ecx, [r8 + 4*r8]
	inc r8
	add cl, 5
	mov r14d, 23
	xor edx, edx
	shld rdx, r14, cl
	shl r14, cl
	xor r9d, r9d
	test cl, 64
	cmovne rdx, r14
	cmovne r14, r9
	or rdx, rsi
	or r14, rdi
	mov rdi, r14
	mov rsi, rdx
	mov qword ptr [rsp + 48], r8
.Ltmp28776:
	cmp rax, r10
	je .LBB234_49
.LBB234_40:
	mov rcx, qword ptr [rsp + 48]
	lea r8, [rcx - 24]
	lea rax, [rcx + 4*rcx]
	add rax, 5
	sub r10, qword ptr [rsp + 24]
	sub r10, rcx
	add r10, 23
	xor r9d, r9d
	mov r14, rdi
	mov rdx, rsi
	.p2align	4
.LBB234_41:
	xor esi, esi
	mov ecx, eax
	shld rsi, r10, cl
	mov rdi, r10
	shl rdi, cl
	test al, 64
	cmovne rsi, rdi
	cmovne rdi, r9
	or rsi, rdx
	or rdi, r14
	lea r14, [r10 - 1]
	lea ecx, [rax + 5]
	xor edx, edx
	shld rdx, r14, cl
	shl r14, cl
	test cl, 64
	cmovne rdx, r14
	cmovne r14, r9
	or rdx, rsi
	or r14, rdi
	add rax, 10
	add r10, -2
	add r8, 2
	jne .LBB234_41
	jmp .LBB234_49
.Ltmp28780:
.LBB234_42:
	mov rax, qword ptr [rsp + 24]
	inc rax
	cmp qword ptr [rsp], rax
	mov r13, qword ptr [rsp + 104]
	mov r12, qword ptr [rsp + 56]
	jne .LBB234_44
	xor esi, esi
	mov rdi, qword ptr [rsp + 48]
	xor edx, edx
	jmp .LBB234_46
.LBB234_44:
	xor r8d, r8d
	mov r14, qword ptr [rsp + 48]
	mov r9, r14
	and r9, -2
	mov eax, 5
	xor esi, esi
	xor edx, edx
	.p2align	4
.LBB234_45:
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
	lea r14d, [rax + 5]
	mov ecx, r14d
	and cl, 62
	xor esi, esi
	shld rsi, rdi, cl
	shl rdi, cl
	add rdx, 2
	test r14b, 64
	cmovne rsi, rdi
	cmovne rdi, r8
	or rsi, r10
	or rdi, r11
	add rax, 10
	mov r14, rdi
	cmp r9, rdx
	jne .LBB234_45
.LBB234_46:
	test byte ptr [rsp + 48], 1
	je .LBB234_48
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
.LBB234_48:
	mov r14, rdi
	mov rdx, rsi
	cmp qword ptr [rsp + 48], 23
	mov rax, qword ptr [rsp]
	jbe .LBB234_24
.Ltmp28788:
.LBB234_49:
	mov rdi, r12
	add rdi, 64
.Ltmp28789:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC]
.Ltmp28790:
.Ltmp28569:
	mov rsi, r14
	call rax
.Ltmp28791:
	mov rsi, qword ptr [rsp + 8]
.Ltmp28792:
	and rsi, -32
.Ltmp28793:
	or rsi, qword ptr [rsp + 24]
.Ltmp28794:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC]
	mov rdi, qword ptr [rsp + 40]
	mov rdx, qword ptr [rsp + 80]
.Ltmp28796:
	call rax
.Ltmp28797:
.Ltmp28572:
	shr r14d, 5
.Ltmp28798:
	and r14d, 31
.Ltmp28799:
	cmp r14d, 24
	jae .LBB234_223
.Ltmp28800:
	mov rax, qword ptr [r12 + 8*r14 + 128]
.Ltmp28801:
	mov qword ptr [rsp + 40], rax
.Ltmp28802:
	lea r12, [r13 + 960]
.Ltmp28803:
	mov cl, 1
.Ltmp28804:
	xor eax, eax
	lock cmpxchg	byte ptr [r13 + 960], cl
.Ltmp28805:
	jne .LBB234_210
.Ltmp28806:
.LBB234_53:
	mov r14, qword ptr [r13 + 984]
.Ltmp28807:
	cmp r14, qword ptr [r13 + 968]
	jne .LBB234_55
.Ltmp28808:
.Ltmp28578:
	lea rdi, [r13 + 968]
.Ltmp28809:
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp28810:
.Ltmp28579:
.LBB234_55:
	mov rax, qword ptr [r13 + 976]
	mov rcx, qword ptr [rsp + 56]
.Ltmp28812:
	mov qword ptr [rax + 8*r14], rcx
.Ltmp28813:
	inc r14
.Ltmp28814:
	mov qword ptr [r13 + 984], r14
	xor ecx, ecx
.Ltmp28815:
	mov al, 1
	lock cmpxchg	byte ptr [r13 + 960], cl
.Ltmp28816:
	jne .LBB234_211
.Ltmp28817:
.LBB234_56:
	mov r14, qword ptr [rsp + 72]
	mov r12, qword ptr [rsp + 56]
.Ltmp28818:
	mov rcx, qword ptr [r14 + 544]
.Ltmp28819:
	test cl, 1
	je .LBB234_58
	jmp .LBB234_57
.Ltmp28820:
	.p2align	4
.LBB234_59:
	pause
.Ltmp28822:
	mov rcx, qword ptr [r14 + 544]
.Ltmp28823:
	test cl, 1
	je .LBB234_58
.Ltmp28824:
.LBB234_57:
	mov rdi, r14
	call masstree::leaf24::LeafNode24<S>::wait_for_split
.Ltmp28825:
	mov rcx, qword ptr [r14 + 544]
.Ltmp28826:
	test cl, 1
	jne .LBB234_57
.Ltmp28827:
.LBB234_58:
	lea rdx, [rcx + 1]
.Ltmp28828:
	mov rax, rcx
	lock cmpxchg	qword ptr [r14 + 544], rdx
.Ltmp28829:
	jne .LBB234_59
.Ltmp28830:
	mov qword ptr [r12 + 552], r14
.Ltmp28831:
	mov qword ptr [r12 + 544], rcx
.Ltmp28832:
	test rcx, rcx
.Ltmp28833:
	je .LBB234_62
.Ltmp28834:
	mov qword ptr [rcx + 552], r12
.Ltmp28835:
.LBB234_62:
	#MEMBARRIER
	mov qword ptr [r14 + 544], r12
	mov esi, dword ptr [rsp + 92]
	mov byte ptr [rsp + 22], sil
	mov eax, dword ptr [rsp + 36]
	mov byte ptr [rsp + 23], al
	lea rax, [r13 + 992]
	mov qword ptr [rsp], rax
	lea rax, [r13 + 1000]
	mov qword ptr [rsp + 112], rax
	mov eax, 1
	mov edx, 560
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.202]
	mov qword ptr [rsp + 176], rcx
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.206]
	mov qword ptr [rsp + 96], rcx
	mov cl, 1
	jmp .LBB234_64
.Ltmp28837:
	.p2align	4
.LBB234_63:
	mov rcx, qword ptr [rsp + 72]
.Ltmp28838:
	mov qword ptr [r9 + rcx], rax
.Ltmp28839:
	mov eax, dword ptr [r9]
.Ltmp28840:
	add eax, 512
	and eax, -1342177792
.Ltmp28841:
	add r8d, 512
	and r8d, -1342177792
.Ltmp28842:
	cmp qword ptr [rsp + 120], rbx
.Ltmp28843:
	#MEMBARRIER
	mov dword ptr [r9], eax
	mov rax, qword ptr [rsp + 64]
.Ltmp28844:
	mov dword ptr [rax], r8d
.Ltmp28845:
	sete byte ptr [rsp + 22]
	mov eax, dword ptr [rsp + 36]
	mov byte ptr [rsp + 23], al
.Ltmp28846:
	mov rdx, qword ptr [rsp + 48]
.Ltmp28847:
	lea rax, [rdx + 1]
	xor ecx, ecx
	mov r14, rbx
	cmp rdx, 63
	mov esi, dword ptr [rsp + 92]
	mov edx, 560
	ja .LBB234_230
.Ltmp28848:
.LBB234_64:
	mov qword ptr [rsp + 8], r12
	mov r9d, r15d
.Ltmp28849:
	test cl, 1
	mov edi, 264
	cmovne rdi, rdx
	mov qword ptr [rsp + 72], rdi
	mov qword ptr [rsp + 64], rbx
.Ltmp28850:
	not sil
	mov rdi, rax
	xor edx, edx
	mov r12, r14
	mov r8d, esi
	mov dword ptr [rsp + 24], r15d
.Ltmp28851:
	jmp .LBB234_68
.Ltmp28852:
	.p2align	4
.LBB234_65:
	mov rax, qword ptr [rsp + 72]
.Ltmp28853:
	mov rax, qword ptr [r14 + rax]
.Ltmp28854:
	cmp rax, rbx
	je .LBB234_79
	xor edx, edx
.Ltmp28856:
.LBB234_67:
	lea eax, [r15 + 512]
	mov esi, r15d
	add esi, 8
	test r15b, 4
	mov edi, -1342177792
	mov r10d, -268435464
	cmove edi, r10d
	cmovne esi, eax
	and esi, edi
	mov dword ptr [rbx], esi
.Ltmp28857:
	pause
.Ltmp28858:
	mov rax, qword ptr [rsp + 48]
.Ltmp28859:
	cmp rax, 63
	lea rdi, [rax + 1]
.Ltmp28860:
	ja .LBB234_229
.Ltmp28861:
.LBB234_68:
	mov rax, qword ptr [rsp + 72]
.Ltmp28862:
	mov rbx, qword ptr [r14 + rax]
.Ltmp28863:
	test rbx, rbx
.Ltmp28864:
	jne .LBB234_70
.Ltmp28865:
	test byte ptr [rsp + 36], 1
	jne .LBB234_160
.Ltmp28866:
.LBB234_70:
	test rbx, rbx
	setne al
.Ltmp28867:
	or al, r8b
	test al, 1
	je .LBB234_167
.Ltmp28868:
	test rbx, rbx
.Ltmp28869:
	je .LBB234_228
.Ltmp28870:
	mov qword ptr [rsp + 48], rdi
	xor edi, edi
	jmp .LBB234_74
.Ltmp28871:
	.p2align	4
.LBB234_73:
	and edi, 7
	lea edi, [2*rdi + 1]
.LBB234_74:
	mov r15d, dword ptr [rbx]
	test r15b, 1
	jne .LBB234_76
	mov esi, r15d
	or esi, 3
	mov eax, r15d
	lock cmpxchg	dword ptr [rbx], esi
	je .LBB234_65
.LBB234_76:
	xor eax, eax
	.p2align	4
.LBB234_77:
	mov esi, eax
	pause
	cmp eax, edi
	adc eax, 0
	cmp esi, edi
	jae .LBB234_73
	cmp eax, edi
	jbe .LBB234_77
	jmp .LBB234_73
.Ltmp28877:
	.p2align	4
.LBB234_79:
	movzx eax, byte ptr [rbx + 4]
.Ltmp28879:
	movzx r14d, al
.Ltmp28880:
	test r14b, r14b
.Ltmp28881:
	je .LBB234_86
	xor eax, eax
	jmp .LBB234_81
.Ltmp28883:
	.p2align	4
.LBB234_83:
	mov rdi, qword ptr [rbx + 256]
.Ltmp28884:
	cmp rdi, r12
.Ltmp28885:
	je .LBB234_89
.Ltmp28886:
.LBB234_84:
	inc rax
.Ltmp28887:
	cmp r14, rax
.Ltmp28888:
	je .LBB234_85
.Ltmp28889:
.LBB234_81:
	cmp rax, 15
	jae .LBB234_83
.Ltmp28890:
	mov rdi, qword ptr [rbx + 8*rax + 136]
.Ltmp28891:
	cmp rdi, r12
.Ltmp28892:
	jne .LBB234_84
	jmp .LBB234_89
.Ltmp28893:
.LBB234_85:
	dec rax
.Ltmp28894:
	cmp rax, 14
	jae .LBB234_87
.Ltmp28895:
.LBB234_86:
	mov rax, qword ptr [rbx + 8*r14 + 136]
.Ltmp28896:
	cmp rax, r12
.Ltmp28897:
	jne .LBB234_88
	jmp .LBB234_90
.Ltmp28898:
.LBB234_87:
	mov rax, qword ptr [rbx + 256]
.Ltmp28899:
	cmp rax, r12
.Ltmp28900:
	je .LBB234_90
.Ltmp28901:
.LBB234_88:
	inc rdx
	cmp rdx, 16
	mov r14, r12
	jbe .LBB234_67
	jmp .LBB234_194
	.p2align	4
.LBB234_89:
	mov r14, rax
.Ltmp28905:
.LBB234_90:
	movzx eax, byte ptr [rbx + 4]
.Ltmp28906:
	cmp al, 14
.Ltmp28907:
	jbe .LBB234_195
.Ltmp28908:
	or dword ptr [rbx], 4
.Ltmp28909:
	#MEMBARRIER
	mov rcx, qword ptr [r13 + 1024]
.Ltmp28910:
	cmp rcx, rbx
	sete al
	mov dword ptr [rsp + 92], eax
.Ltmp28911:
	mov qword ptr [rsp + 120], rcx
.Ltmp28912:
	je .LBB234_94
.Ltmp28913:
	mov rax, qword ptr [rbx + 264]
.Ltmp28914:
	test rax, rax
.Ltmp28915:
	je .LBB234_95
.LBB234_94:
	mov dword ptr [rsp + 36], 0
	jmp .LBB234_96
.Ltmp28917:
.LBB234_95:
	mov eax, dword ptr [rbx]
.Ltmp28918:
	shr eax, 30
.Ltmp28919:
	and al, 1
	mov dword ptr [rsp + 36], eax
.Ltmp28920:
.LBB234_96:
	or r15d, 7
.Ltmp28921:
	mov r13d, dword ptr [rbx + 8]
.Ltmp28922:
	mov r12d, dword ptr [rbx]
.Ltmp28923:
	lea rax, [rsp + 196]
	xorps xmm0, xmm0
.Ltmp28924:
	movups xmmword ptr [rax + 96], xmm0
	movups xmmword ptr [rax + 80], xmm0
	movups xmmword ptr [rax + 64], xmm0
	movups xmmword ptr [rax + 48], xmm0
	movups xmmword ptr [rax + 32], xmm0
	movups xmmword ptr [rax + 16], xmm0
	movups xmmword ptr [rax], xmm0
	mov qword ptr [rax + 112], 0
.Ltmp28925:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp28926:
	test rax, rax
	je .LBB234_208
.Ltmp28927:
	mov qword ptr [rsp + 80], rax
	and r12d, -2147483648
	or r12d, 5
.Ltmp28928:
	mov dword ptr [rax], r12d
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r13d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
	xorps xmm0, xmm0
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rsp + 56], rax
	mov qword ptr [rax + 264], 0
.Ltmp28929:
	xor eax, eax
	mov rcx, qword ptr [rsp]
	mov dl, 1
	lock cmpxchg	byte ptr [rcx], dl
	mov r13, qword ptr [rsp + 104]
.Ltmp28930:
	jne .LBB234_158
.Ltmp28931:
.LBB234_98:
	mov r12, qword ptr [r13 + 1016]
.Ltmp28932:
	cmp r12, qword ptr [r13 + 1000]
	jne .LBB234_100
.Ltmp28589:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp28590:
.LBB234_100:
	mov rdx, qword ptr [rsp + 104]
.Ltmp28935:
	mov rax, qword ptr [rdx + 1008]
	mov rcx, qword ptr [rsp + 56]
.Ltmp28937:
	mov qword ptr [rax + 8*r12], rcx
.Ltmp28938:
	inc r12
.Ltmp28939:
	mov qword ptr [rdx + 1016], r12
.Ltmp28940:
	mov al, 1
	xor ecx, ecx
	lock cmpxchg	byte ptr [rdx + 992], cl
.Ltmp28941:
	jne .LBB234_159
.Ltmp28942:
.LBB234_101:
	add qword ptr [rsp + 80], 136
.Ltmp28943:
	cmp r14, 9
	setae al
	cmp r14, 8
	sbb al, 0
	mov r8d, dword ptr [rsp + 24]
	mov r12, qword ptr [rsp + 56]
.Ltmp28944:
	je .LBB234_118
	movzx eax, al
	cmp eax, 1
	mov r13, qword ptr [rsp + 104]
	jne .LBB234_120
.Ltmp28946:
	mov rax, r14
.Ltmp28947:
	mov rcx, qword ptr [rbx + 208]
	mov rdx, qword ptr [rsp + 80]
.Ltmp28949:
	mov qword ptr [rdx], rcx
.Ltmp28950:
	add rax, -9
.Ltmp28951:
	je .LBB234_109
.Ltmp28952:
	mov rcx, qword ptr [rbx + 88]
.Ltmp28953:
	mov qword ptr [r12 + 16], rcx
.Ltmp28954:
	mov rcx, qword ptr [rbx + 216]
.Ltmp28955:
	mov qword ptr [r12 + 144], rcx
.Ltmp28956:
	cmp rax, 1
.Ltmp28957:
	je .LBB234_109
.Ltmp28958:
	mov rcx, qword ptr [rbx + 96]
.Ltmp28959:
	mov qword ptr [r12 + 24], rcx
.Ltmp28960:
	mov rcx, qword ptr [rbx + 224]
.Ltmp28961:
	mov qword ptr [r12 + 152], rcx
.Ltmp28962:
	cmp rax, 2
.Ltmp28963:
	je .LBB234_109
.Ltmp28964:
	mov rcx, qword ptr [rbx + 104]
.Ltmp28965:
	mov qword ptr [r12 + 32], rcx
.Ltmp28966:
	mov rcx, qword ptr [rbx + 232]
.Ltmp28967:
	mov qword ptr [r12 + 160], rcx
.Ltmp28968:
	cmp rax, 3
.Ltmp28969:
	je .LBB234_109
.Ltmp28970:
	mov rcx, qword ptr [rbx + 112]
.Ltmp28971:
	mov qword ptr [r12 + 40], rcx
.Ltmp28972:
	mov rcx, qword ptr [rbx + 240]
.Ltmp28973:
	mov qword ptr [r12 + 168], rcx
.Ltmp28974:
	cmp rax, 4
.Ltmp28975:
	je .LBB234_109
.Ltmp28976:
	mov rcx, qword ptr [rbx + 120]
.Ltmp28977:
	mov qword ptr [r12 + 48], rcx
.Ltmp28978:
	mov rcx, qword ptr [rbx + 248]
.Ltmp28979:
	mov qword ptr [r12 + 176], rcx
.Ltmp28980:
	cmp rax, 5
.Ltmp28981:
	jne .LBB234_156
.Ltmp28982:
.LBB234_109:
	mov rax, qword ptr [rsp + 40]
.Ltmp28983:
	mov qword ptr [r12 + 8*r14 - 56], rax
.Ltmp28984:
	mov rax, qword ptr [rsp + 8]
.Ltmp28985:
	mov qword ptr [r12 + 8*r14 + 72], rax
	inc r14
.Ltmp28986:
	jmp .LBB234_111
.Ltmp28987:
	.p2align	4
.LBB234_110:
	mov qword ptr [r12 + 8*r14 + 72], rax
.Ltmp28988:
	inc r14
.Ltmp28989:
	cmp r14, 16
.Ltmp28990:
	je .LBB234_117
.Ltmp28991:
.LBB234_111:
	lea rax, [r14 - 1]
	cmp rax, 14
	ja .LBB234_225
.Ltmp28992:
	mov rcx, qword ptr [rbx + 8*r14 + 8]
.Ltmp28993:
	lea rax, [r14 - 9]
	cmp rax, 14
	ja .LBB234_224
.Ltmp28994:
	mov qword ptr [r12 + 8*r14 - 56], rcx
.Ltmp28995:
	cmp r14, 15
	jae .LBB234_115
.Ltmp28996:
	mov rax, qword ptr [rbx + 8*r14 + 136]
.Ltmp28997:
	lea rcx, [r14 - 8]
.Ltmp28998:
	cmp rcx, 15
	jb .LBB234_110
	jmp .LBB234_116
.Ltmp28999:
	.p2align	4
.LBB234_115:
	mov rax, qword ptr [rbx + 256]
.Ltmp29001:
	lea rcx, [r14 - 8]
.Ltmp29002:
	cmp rcx, 15
	jb .LBB234_110
.Ltmp29003:
.LBB234_116:
	mov qword ptr [r12 + 256], rax
.Ltmp29004:
	inc r14
.Ltmp29005:
	cmp r14, 16
.Ltmp29006:
	jne .LBB234_111
.Ltmp29007:
.LBB234_117:
	mov byte ptr [r12 + 4], 7
.Ltmp29008:
	mov rax, qword ptr [rbx + 80]
	mov qword ptr [rsp + 40], rax
.Ltmp29009:
	mov cl, 8
	mov rax, r12
	mov r9, qword ptr [rsp + 8]
	jmp .LBB234_119
.Ltmp29011:
.LBB234_118:
	mov r9, qword ptr [rsp + 8]
.Ltmp29012:
	mov qword ptr [r12 + 136], r9
.Ltmp29013:
	mov rax, qword ptr [rbx + 80]
.Ltmp29014:
	mov qword ptr [r12 + 16], rax
.Ltmp29015:
	mov rax, qword ptr [rbx + 208]
.Ltmp29016:
	mov qword ptr [r12 + 144], rax
.Ltmp29017:
	mov rax, qword ptr [rbx + 88]
.Ltmp29018:
	mov qword ptr [r12 + 24], rax
.Ltmp29019:
	mov rax, qword ptr [rbx + 216]
.Ltmp29020:
	mov qword ptr [r12 + 152], rax
.Ltmp29021:
	mov rax, qword ptr [rbx + 96]
.Ltmp29022:
	mov qword ptr [r12 + 32], rax
.Ltmp29023:
	mov rax, qword ptr [rbx + 224]
.Ltmp29024:
	mov qword ptr [r12 + 160], rax
.Ltmp29025:
	mov rax, qword ptr [rbx + 104]
.Ltmp29026:
	mov qword ptr [r12 + 40], rax
.Ltmp29027:
	mov rax, qword ptr [rbx + 232]
.Ltmp29028:
	mov qword ptr [r12 + 168], rax
.Ltmp29029:
	mov rax, qword ptr [rbx + 112]
.Ltmp29030:
	mov qword ptr [r12 + 48], rax
.Ltmp29031:
	mov rax, qword ptr [rbx + 240]
.Ltmp29032:
	mov qword ptr [r12 + 176], rax
.Ltmp29033:
	mov rax, qword ptr [rbx + 120]
.Ltmp29034:
	mov qword ptr [r12 + 56], rax
.Ltmp29035:
	mov rax, qword ptr [rbx + 248]
.Ltmp29036:
	mov qword ptr [r12 + 184], rax
.Ltmp29037:
	mov rax, qword ptr [rbx + 128]
.Ltmp29038:
	mov qword ptr [r12 + 64], rax
.Ltmp29039:
	mov rax, qword ptr [rbx + 256]
.Ltmp29040:
	mov qword ptr [r12 + 192], rax
.Ltmp29041:
	mov byte ptr [r12 + 4], 7
	mov cl, 8
	mov rax, r12
.Ltmp29043:
	mov r13, qword ptr [rsp + 104]
.Ltmp29044:
.LBB234_119:
	mov r10, qword ptr [rsp + 80]
	jmp .LBB234_132
.Ltmp29045:
.LBB234_120:
	mov rax, qword ptr [rbx + 200]
.Ltmp29046:
	mov qword ptr [r12 + 136], rax
.Ltmp29047:
	mov rax, qword ptr [rbx + 80]
.Ltmp29048:
	mov qword ptr [r12 + 16], rax
.Ltmp29049:
	mov rax, qword ptr [rbx + 208]
.Ltmp29050:
	mov qword ptr [r12 + 144], rax
.Ltmp29051:
	mov rax, qword ptr [rbx + 88]
.Ltmp29052:
	mov qword ptr [r12 + 24], rax
.Ltmp29053:
	mov rax, qword ptr [rbx + 216]
.Ltmp29054:
	mov qword ptr [r12 + 152], rax
.Ltmp29055:
	mov rax, qword ptr [rbx + 96]
.Ltmp29056:
	mov qword ptr [r12 + 32], rax
.Ltmp29057:
	mov rax, qword ptr [rbx + 224]
.Ltmp29058:
	mov qword ptr [r12 + 160], rax
.Ltmp29059:
	mov rax, qword ptr [rbx + 104]
.Ltmp29060:
	mov qword ptr [r12 + 40], rax
.Ltmp29061:
	mov rax, qword ptr [rbx + 232]
.Ltmp29062:
	mov qword ptr [r12 + 168], rax
.Ltmp29063:
	mov rax, qword ptr [rbx + 112]
.Ltmp29064:
	mov qword ptr [r12 + 48], rax
.Ltmp29065:
	mov rax, qword ptr [rbx + 240]
.Ltmp29066:
	mov qword ptr [r12 + 176], rax
.Ltmp29067:
	mov rax, qword ptr [rbx + 120]
.Ltmp29068:
	mov qword ptr [r12 + 56], rax
.Ltmp29069:
	mov rax, qword ptr [rbx + 248]
.Ltmp29070:
	mov qword ptr [r12 + 184], rax
.Ltmp29071:
	mov rax, qword ptr [rbx + 128]
.Ltmp29072:
	mov qword ptr [r12 + 64], rax
.Ltmp29073:
	mov rax, qword ptr [rbx + 256]
.Ltmp29074:
	mov qword ptr [r12 + 192], rax
.Ltmp29075:
	mov byte ptr [r12 + 4], 7
.Ltmp29076:
	mov rdx, qword ptr [rbx + 72]
.Ltmp29077:
	mov byte ptr [rbx + 4], 7
.Ltmp29078:
	movzx eax, byte ptr [rbx + 4]
.Ltmp29079:
	movzx ecx, al
.Ltmp29080:
	cmp r14, rcx
.Ltmp29081:
	jae .LBB234_127
.Ltmp29082:
	mov rax, rcx
	jmp .LBB234_123
.Ltmp29083:
	.p2align	4
.LBB234_122:
	mov qword ptr [rbx + 8*rsi + 152], rdi
	mov rax, rsi
.Ltmp29085:
	cmp r14, rsi
.Ltmp29086:
	jae .LBB234_127
.Ltmp29087:
.LBB234_123:
	cmp rax, 16
	jae .LBB234_232
.Ltmp29088:
	lea rsi, [rax - 1]
.Ltmp29089:
	mov rdi, qword ptr [rbx + 8*rsi + 16]
.Ltmp29090:
	cmp rax, 15
	je .LBB234_235
.Ltmp29091:
	mov qword ptr [rbx + 8*rsi + 24], rdi
.Ltmp29092:
	mov rdi, qword ptr [rbx + 8*rsi + 144]
.Ltmp29093:
	cmp rax, 14
	jb .LBB234_122
.Ltmp29094:
	mov qword ptr [rbx + 256], rdi
	mov rax, rsi
.Ltmp29096:
	cmp r14, rsi
.Ltmp29097:
	jb .LBB234_123
.Ltmp29098:
.LBB234_127:
	cmp r14, 14
	ja .LBB234_238
	mov rax, qword ptr [rsp + 40]
.Ltmp29100:
	mov qword ptr [rbx + 8*r14 + 16], rax
.Ltmp29101:
	jne .LBB234_130
	mov r9, qword ptr [rsp + 8]
.Ltmp29103:
	mov qword ptr [rbx + 256], r9
	jmp .LBB234_131
.Ltmp29104:
.LBB234_130:
	mov r9, qword ptr [rsp + 8]
.Ltmp29105:
	mov qword ptr [rbx + 8*r14 + 144], r9
.Ltmp29106:
.LBB234_131:
	mov r10, qword ptr [rsp + 80]
	#MEMBARRIER
	inc cl
.Ltmp29108:
	mov rax, rbx
	mov qword ptr [rsp + 40], rdx
.Ltmp29109:
.LBB234_132:
	mov byte ptr [rbx + 4], cl
.Ltmp29110:
	mov ecx, dword ptr [rbx + 8]
	mov dword ptr [r12 + 8], ecx
	test ecx, ecx
	je .LBB234_148
.Ltmp29111:
	movzx ecx, byte ptr [r12 + 4]
.Ltmp29112:
	movzx ecx, cl
.Ltmp29113:
	xor esi, esi
.Ltmp29114:
	xor edx, edx
.Ltmp29115:
	.p2align	4
.LBB234_134:
	cmp rsi, rcx
.Ltmp29117:
	adc rdx, 0
.Ltmp29118:
	cmp rsi, 15
	jae .LBB234_136
.Ltmp29119:
	mov rdi, qword ptr [r10 + 8*rsi]
.Ltmp29120:
	test rdi, rdi
.Ltmp29121:
	jne .LBB234_137
	jmp .LBB234_138
.Ltmp29122:
	.p2align	4
.LBB234_136:
	mov rdi, qword ptr [r12 + 256]
.Ltmp29124:
	test rdi, rdi
.Ltmp29125:
	je .LBB234_138
.Ltmp29126:
.LBB234_137:
	mov qword ptr [rdi + 264], r12
.Ltmp29127:
.LBB234_138:
	cmp rsi, rcx
.Ltmp29128:
	jae .LBB234_140
	mov rsi, rdx
	cmp rdx, rcx
	jbe .LBB234_134
.Ltmp29130:
.LBB234_140:
	cmp dword ptr [rbx + 8], 0
.Ltmp29131:
	movzx ecx, byte ptr [r12 + 4]
.Ltmp29132:
	movzx ecx, cl
.Ltmp29133:
	je .LBB234_149
	xor esi, esi
.Ltmp29135:
	xor edx, edx
.Ltmp29136:
	.p2align	4
.LBB234_142:
	cmp rsi, rcx
.Ltmp29138:
	adc rdx, 0
.Ltmp29139:
	cmp rsi, 15
	jae .LBB234_144
.Ltmp29140:
	mov rdi, qword ptr [r10 + 8*rsi]
.Ltmp29141:
	test rdi, rdi
.Ltmp29142:
	jne .LBB234_145
	jmp .LBB234_146
.Ltmp29143:
	.p2align	4
.LBB234_144:
	mov rdi, qword ptr [r12 + 256]
.Ltmp29145:
	test rdi, rdi
.Ltmp29146:
	je .LBB234_146
.Ltmp29147:
.LBB234_145:
	mov qword ptr [rdi + 264], r12
.Ltmp29148:
.LBB234_146:
	cmp rsi, rcx
.Ltmp29149:
	jae .LBB234_63
	mov rsi, rdx
	cmp rdx, rcx
	jbe .LBB234_142
	jmp .LBB234_63
.Ltmp29151:
.LBB234_148:
	movzx ecx, byte ptr [r12 + 4]
.Ltmp29152:
	movzx ecx, cl
.Ltmp29153:
.LBB234_149:
	xor esi, esi
.Ltmp29154:
	xor edx, edx
.Ltmp29155:
	.p2align	4
.LBB234_150:
	cmp rsi, rcx
.Ltmp29157:
	adc rdx, 0
.Ltmp29158:
	cmp rsi, 15
	jae .LBB234_152
.Ltmp29159:
	mov rdi, qword ptr [r10 + 8*rsi]
.Ltmp29160:
	test rdi, rdi
.Ltmp29161:
	jne .LBB234_153
	jmp .LBB234_154
.Ltmp29162:
	.p2align	4
.LBB234_152:
	mov rdi, qword ptr [r12 + 256]
.Ltmp29164:
	test rdi, rdi
.Ltmp29165:
	je .LBB234_154
.Ltmp29166:
.LBB234_153:
	mov qword ptr [rdi + 560], r12
.Ltmp29167:
.LBB234_154:
	cmp rsi, rcx
.Ltmp29168:
	jae .LBB234_63
	mov rsi, rdx
	cmp rdx, rcx
	jbe .LBB234_150
	jmp .LBB234_63
.Ltmp29170:
.LBB234_156:
	mov rcx, qword ptr [rbx + 128]
.Ltmp29171:
	mov qword ptr [r12 + 56], rcx
.Ltmp29172:
	mov rcx, qword ptr [rbx + 256]
.Ltmp29173:
	mov qword ptr [r12 + 184], rcx
.Ltmp29174:
	cmp rax, 6
.Ltmp29175:
	jne .LBB234_245
.Ltmp29176:
	mov rax, qword ptr [rsp + 40]
.Ltmp29177:
	mov qword ptr [r12 + 64], rax
.Ltmp29178:
	mov rax, qword ptr [rsp + 8]
.Ltmp29179:
	mov qword ptr [r12 + 8*r14 + 72], rax
	jmp .LBB234_117
.Ltmp29180:
.LBB234_158:
.Ltmp28587:
	mov rdi, qword ptr [rsp]
.Ltmp29181:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB234_98
.Ltmp29182:
.LBB234_159:
	mov rdi, qword ptr [rsp]
.Ltmp29183:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28596:
	jmp .LBB234_101
.Ltmp29184:
.LBB234_160:
	test cl, 1
	je .LBB234_175
.Ltmp29185:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp29186:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp29187:
	test rax, rax
	je .LBB234_212
.Ltmp29188:
	mov rbx, rax
.Ltmp29189:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp29190:
	xorps xmm0, xmm0
.Ltmp29191:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp29192:
	mov qword ptr [rax + 136], r14
	mov rax, qword ptr [rsp + 40]
.Ltmp29194:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 8]
.Ltmp29196:
	mov qword ptr [rbx + 144], rax
.Ltmp29197:
	mov byte ptr [rbx + 4], 1
.Ltmp29198:
	lock or	dword ptr [rbx], 1073741824
	mov cl, 1
.Ltmp29199:
	xor eax, eax
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29200:
	jne .LBB234_213
.Ltmp29201:
.LBB234_163:
	mov r14, qword ptr [r13 + 1016]
.Ltmp29202:
	cmp r14, qword ptr [r13 + 1000]
	jne .LBB234_165
.Ltmp28637:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp28638:
.Ltmp29204:
.LBB234_165:
	mov rax, qword ptr [r13 + 1008]
.Ltmp29205:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp29206:
	inc r14
.Ltmp29207:
	mov qword ptr [r13 + 1016], r14
	xor ecx, ecx
.Ltmp29208:
	mov al, 1
	lock cmpxchg	byte ptr [r13 + 992], cl
.Ltmp29209:
	jne .LBB234_215
.Ltmp29210:
.LBB234_166:
	#MEMBARRIER
	mov qword ptr [r12 + 560], rbx
	mov rcx, qword ptr [rsp + 8]
.Ltmp29212:
	mov qword ptr [rcx + 560], rbx
.Ltmp29213:
	lock and	dword ptr [r12], -1073741825
	mov r12, rcx
.Ltmp29215:
	jmp .LBB234_181
.Ltmp29216:
.LBB234_167:
	test cl, 1
	je .LBB234_182
.Ltmp29217:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp29218:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp29219:
	test rax, rax
	je .LBB234_212
.Ltmp29220:
	mov rbx, rax
.Ltmp29221:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], 0
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp29222:
	xorps xmm0, xmm0
.Ltmp29223:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp29224:
	lock or	dword ptr [rax], 1073741824
.Ltmp29225:
	mov qword ptr [rax + 136], r14
	mov rax, qword ptr [rsp + 40]
.Ltmp29227:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 8]
.Ltmp29229:
	mov qword ptr [rbx + 144], rax
.Ltmp29230:
	mov byte ptr [rbx + 4], 1
	mov cl, 1
.Ltmp29231:
	xor eax, eax
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29232:
	jne .LBB234_217
.Ltmp29233:
.LBB234_170:
	mov r14, qword ptr [r13 + 1016]
.Ltmp29234:
	cmp r14, qword ptr [r13 + 1000]
	jne .LBB234_172
.Ltmp28615:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp28616:
.Ltmp29236:
.LBB234_172:
	mov rax, qword ptr [r13 + 1008]
.Ltmp29237:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp29238:
	inc r14
.Ltmp29239:
	mov qword ptr [r13 + 1016], r14
	xor ecx, ecx
.Ltmp29240:
	mov al, 1
	lock cmpxchg	byte ptr [r13 + 992], cl
.Ltmp29241:
	jne .LBB234_219
.Ltmp29242:
.LBB234_173:
	mov qword ptr [rsp + 128], r12
.Ltmp29243:
	mov rax, r12
	lock cmpxchg	qword ptr [r13 + 1024], rbx
	mov rdx, qword ptr [rsp + 8]
.Ltmp29244:
	jne .LBB234_240
.Ltmp29245:
	#MEMBARRIER
	mov qword ptr [r12 + 560], rbx
.Ltmp29246:
	mov qword ptr [rdx + 560], rbx
.Ltmp29247:
	jmp .LBB234_189
.Ltmp29248:
.LBB234_175:
	mov r15, r14
.Ltmp29249:
	mov r14d, dword ptr [r14 + 8]
.Ltmp29250:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp29251:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp29252:
	test rax, rax
	je .LBB234_212
.Ltmp29253:
	mov rbx, rax
.Ltmp29254:
	inc r14d
.Ltmp29255:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r14d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp29256:
	xorps xmm0, xmm0
.Ltmp29257:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp29258:
	mov qword ptr [rax + 136], r15
	mov rax, qword ptr [rsp + 40]
.Ltmp29260:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 8]
.Ltmp29262:
	mov qword ptr [rbx + 144], rax
.Ltmp29263:
	mov byte ptr [rbx + 4], 1
.Ltmp29264:
	lock or	dword ptr [rbx], 1073741824
	mov cl, 1
.Ltmp29265:
	xor eax, eax
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29266:
	jne .LBB234_214
.Ltmp29267:
.LBB234_177:
	mov r14, qword ptr [r13 + 1016]
.Ltmp29268:
	cmp r14, qword ptr [r13 + 1000]
	jne .LBB234_179
.Ltmp28627:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp28628:
.Ltmp29270:
.LBB234_179:
	mov rax, qword ptr [r13 + 1008]
.Ltmp29271:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp29272:
	inc r14
.Ltmp29273:
	mov qword ptr [r13 + 1016], r14
	xor ecx, ecx
.Ltmp29274:
	mov al, 1
	lock cmpxchg	byte ptr [r13 + 992], cl
.Ltmp29275:
	jne .LBB234_216
.Ltmp29276:
.LBB234_180:
	#MEMBARRIER
	mov qword ptr [r12 + 264], rbx
	mov rcx, qword ptr [rsp + 8]
.Ltmp29278:
	mov qword ptr [rcx + 264], rbx
.Ltmp29279:
.LBB234_181:
	lock and	dword ptr [r12], -1073741825
.Ltmp29280:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
	mov al, 6
.Ltmp29281:
	mov rcx, qword ptr [rsp + 64]
	mov r9d, dword ptr [rsp + 24]
.Ltmp29282:
	jmp .LBB234_192
.Ltmp29283:
.LBB234_182:
	mov r15, r14
.Ltmp29284:
	mov r14d, dword ptr [r14 + 8]
.Ltmp29285:
	xorps xmm0, xmm0
	movups xmmword ptr [rsp + 292], xmm0
	movups xmmword ptr [rsp + 276], xmm0
	movups xmmword ptr [rsp + 260], xmm0
	movups xmmword ptr [rsp + 244], xmm0
	movups xmmword ptr [rsp + 228], xmm0
	movups xmmword ptr [rsp + 212], xmm0
	movups xmmword ptr [rsp + 196], xmm0
	mov qword ptr [rsp + 308], 0
.Ltmp29286:
	mov edi, 320
	mov esi, 64
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp29287:
	test rax, rax
	je .LBB234_212
.Ltmp29288:
	mov rbx, rax
.Ltmp29289:
	inc r14d
.Ltmp29290:
	mov dword ptr [rax], 0
	mov byte ptr [rax + 4], 0
	mov dword ptr [rax + 8], r14d
	movups xmm0, xmmword ptr [rsp + 192]
	movups xmm1, xmmword ptr [rsp + 208]
	movups xmm2, xmmword ptr [rsp + 224]
	movups xmm3, xmmword ptr [rsp + 240]
	movups xmmword ptr [rax + 12], xmm0
	movups xmmword ptr [rax + 28], xmm1
	movups xmmword ptr [rax + 44], xmm2
	movups xmmword ptr [rax + 60], xmm3
	movups xmm0, xmmword ptr [rsp + 256]
	movups xmmword ptr [rax + 76], xmm0
	movups xmm0, xmmword ptr [rsp + 272]
	movups xmmword ptr [rax + 92], xmm0
	movups xmm0, xmmword ptr [rsp + 288]
	movups xmmword ptr [rax + 108], xmm0
	movups xmm0, xmmword ptr [rsp + 300]
	movups xmmword ptr [rax + 120], xmm0
.Ltmp29291:
	xorps xmm0, xmm0
.Ltmp29292:
	movups xmmword ptr [rax + 136], xmm0
	movups xmmword ptr [rax + 152], xmm0
	movups xmmword ptr [rax + 168], xmm0
	movups xmmword ptr [rax + 184], xmm0
	movups xmmword ptr [rax + 200], xmm0
	movups xmmword ptr [rax + 216], xmm0
	movups xmmword ptr [rax + 232], xmm0
	movups xmmword ptr [rax + 248], xmm0
	mov qword ptr [rax + 264], 0
.Ltmp29293:
	lock or	dword ptr [rax], 1073741824
.Ltmp29294:
	mov qword ptr [rax + 136], r15
	mov rax, qword ptr [rsp + 40]
.Ltmp29296:
	mov qword ptr [rbx + 16], rax
	mov rax, qword ptr [rsp + 8]
.Ltmp29298:
	mov qword ptr [rbx + 144], rax
.Ltmp29299:
	mov byte ptr [rbx + 4], 1
	mov cl, 1
.Ltmp29300:
	xor eax, eax
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29301:
	jne .LBB234_218
.Ltmp29302:
.LBB234_184:
	mov r14, qword ptr [r13 + 1016]
.Ltmp29303:
	cmp r14, qword ptr [r13 + 1000]
	jne .LBB234_186
.Ltmp28605:
	mov rdi, qword ptr [rsp + 112]
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp28606:
.Ltmp29305:
.LBB234_186:
	mov rax, qword ptr [r13 + 1008]
.Ltmp29306:
	mov qword ptr [rax + 8*r14], rbx
.Ltmp29307:
	inc r14
.Ltmp29308:
	mov qword ptr [r13 + 1016], r14
	xor ecx, ecx
.Ltmp29309:
	mov al, 1
	lock cmpxchg	byte ptr [r13 + 992], cl
.Ltmp29310:
	jne .LBB234_220
.Ltmp29311:
.LBB234_187:
	mov qword ptr [rsp + 128], r12
.Ltmp29312:
	mov rax, r12
	lock cmpxchg	qword ptr [r13 + 1024], rbx
	mov rdx, qword ptr [rsp + 8]
.Ltmp29313:
	jne .LBB234_241
.Ltmp29314:
	#MEMBARRIER
	mov qword ptr [r12 + 264], rbx
.Ltmp29315:
	mov qword ptr [rdx + 264], rbx
.Ltmp29316:
.LBB234_189:
	lock and	dword ptr [r12], -1073741825
	mov r9d, dword ptr [rsp + 24]
.Ltmp29317:
	mov eax, dword ptr [rdx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rdx], eax
.Ltmp29318:
.LBB234_190:
	mov al, 6
.LBB234_191:
	mov rcx, qword ptr [rsp + 64]
.Ltmp29320:
.LBB234_192:
	add r9d, 512
	and r9d, -1342177792
	mov dword ptr [rcx], r9d
.Ltmp29321:
.LBB234_193:
	lea rsp, [rbp - 40]
	pop rbx
	pop r12
	pop r13
.Ltmp29322:
	pop r14
	pop r15
	pop rbp
	.cfi_def_cfa rsp, 8
	ret
.Ltmp29323:
.LBB234_194:
	.cfi_def_cfa rbp, 16
	mov rdi, qword ptr [rsp + 8]
.Ltmp29324:
	mov eax, dword ptr [rdi]
	add eax, 512
	and eax, -1342177792
.Ltmp29325:
	mov ecx, r15d
	add ecx, 8
	test r15b, 4
	mov edx, -268435464
.Ltmp29326:
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r15 + 512]
	cmovne ecx, edx
.Ltmp29327:
	#MEMBARRIER
	mov dword ptr [rdi], eax
.Ltmp29328:
	and ecx, esi
	mov dword ptr [rbx], ecx
	mov al, 4
	jmp .LBB234_191
.Ltmp29330:
.LBB234_195:
	movzx eax, byte ptr [rbx + 4]
.Ltmp29331:
	movzx eax, al
.Ltmp29332:
	cmp r14, rax
.Ltmp29333:
	jae .LBB234_202
.Ltmp29334:
	lea r8, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.206]
	mov rcx, rax
	jmp .LBB234_198
.Ltmp29335:
.LBB234_197:
	mov qword ptr [rbx + 8*rdx + 152], rdi
	mov rcx, rdx
.Ltmp29337:
	cmp r14, rdx
.Ltmp29338:
	jae .LBB234_202
.Ltmp29339:
.LBB234_198:
	cmp rcx, 16
	jae .LBB234_236
.Ltmp29340:
	lea rdx, [rcx - 1]
.Ltmp29341:
	mov rdi, qword ptr [rbx + 8*rdx + 16]
.Ltmp29342:
	cmp rcx, 15
	je .LBB234_237
.Ltmp29343:
	mov qword ptr [rbx + 8*rdx + 24], rdi
.Ltmp29344:
	mov rdi, qword ptr [rbx + 8*rdx + 144]
.Ltmp29345:
	cmp rcx, 14
	jb .LBB234_197
.Ltmp29346:
	mov qword ptr [rbx + 256], rdi
	mov rcx, rdx
.Ltmp29348:
	cmp r14, rdx
.Ltmp29349:
	jb .LBB234_198
.Ltmp29350:
.LBB234_202:
	cmp r14, 14
	ja .LBB234_244
	mov rcx, qword ptr [rsp + 40]
.Ltmp29352:
	mov qword ptr [rbx + 8*r14 + 16], rcx
.Ltmp29353:
	jne .LBB234_205
	mov rdi, qword ptr [rsp + 8]
.Ltmp29355:
	mov qword ptr [rbx + 256], rdi
.Ltmp29356:
	jmp .LBB234_206
.LBB234_205:
	mov rdi, qword ptr [rsp + 8]
.Ltmp29358:
	mov qword ptr [rbx + 8*r14 + 144], rdi
.Ltmp29359:
.LBB234_206:
	#MEMBARRIER
	inc al
.Ltmp29360:
	mov byte ptr [rbx + 4], al
	mov rax, qword ptr [rsp + 72]
.Ltmp29362:
	mov qword ptr [rdi + rax], rbx
.Ltmp29363:
	mov eax, dword ptr [rdi]
	add eax, 512
	and eax, -1342177792
.Ltmp29364:
	mov ecx, r15d
	add ecx, 8
	test r15b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	lea edx, [r15 + 512]
	cmovne ecx, edx
.Ltmp29365:
	#MEMBARRIER
	mov dword ptr [rdi], eax
.Ltmp29366:
	and ecx, esi
	mov dword ptr [rbx], ecx
	jmp .LBB234_190
.Ltmp29367:
.LBB234_207:
.Ltmp28564:
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.197]
	mov rdi, rdx
	mov rdx, rax
.Ltmp29368:
	mov r12, qword ptr [rsp + 56]
.Ltmp29369:
	call core::slice::index::slice_index_fail
.Ltmp29370:
.Ltmp28565:
	jmp .LBB234_243
.Ltmp29371:
.LBB234_208:
.Ltmp28600:
	mov edi, 64
	mov esi, 320
	call alloc::alloc::handle_alloc_error
.Ltmp28601:
	jmp .LBB234_243
.Ltmp29372:
.LBB234_209:
.Ltmp28648:
	mov edi, 64
	mov esi, 576
	call alloc::alloc::handle_alloc_error
.Ltmp28649:
	jmp .LBB234_243
.Ltmp29373:
.LBB234_210:
.Ltmp28576:
	mov rdi, r12
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB234_53
.Ltmp29374:
.LBB234_211:
	mov rdi, r12
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28585:
	jmp .LBB234_56
.Ltmp29375:
.LBB234_212:
.Ltmp28645:
	mov edi, 64
	mov esi, 320
	call alloc::alloc::handle_alloc_error
	jmp .LBB234_243
.Ltmp29376:
.LBB234_213:
	mov rdi, qword ptr [rsp]
.Ltmp29377:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB234_163
.Ltmp29378:
.LBB234_214:
	mov rdi, qword ptr [rsp]
.Ltmp29379:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB234_177
.Ltmp29380:
.LBB234_215:
	mov rdi, qword ptr [rsp]
.Ltmp29381:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
	jmp .LBB234_166
.Ltmp29382:
.LBB234_216:
	mov rdi, qword ptr [rsp]
.Ltmp29383:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
	jmp .LBB234_180
.Ltmp29384:
.LBB234_217:
	mov rdi, qword ptr [rsp]
.Ltmp29385:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB234_170
.Ltmp29386:
.LBB234_218:
	mov rdi, qword ptr [rsp]
.Ltmp29387:
	call parking_lot::raw_mutex::RawMutex::lock_slow
	jmp .LBB234_184
.Ltmp29388:
.LBB234_219:
	mov rdi, qword ptr [rsp]
.Ltmp29389:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
	jmp .LBB234_173
.Ltmp29390:
.LBB234_220:
	mov rdi, qword ptr [rsp]
.Ltmp29391:
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28612:
	jmp .LBB234_187
.Ltmp29392:
.LBB234_221:
	mov rdi, r9
.Ltmp29393:
.LBB234_222:
.Ltmp28557:
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
	mov esi, 24
	call core::panicking::panic_bounds_check
.Ltmp28558:
	jmp .LBB234_243
.Ltmp29394:
.LBB234_223:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
	mov qword ptr [rsp + 120], rax
	jmp .LBB234_227
.Ltmp29395:
.LBB234_224:
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.203]
	jmp .LBB234_233
.LBB234_225:
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.202]
	jmp .LBB234_233
.Ltmp29397:
.LBB234_226:
	mov r14d, 24
.Ltmp29398:
.LBB234_227:
.Ltmp28573:
	mov esi, 24
	mov rdi, r14
	mov rdx, qword ptr [rsp + 120]
	mov r12, qword ptr [rsp + 56]
	call core::panicking::panic_bounds_check
.Ltmp28574:
	jmp .LBB234_243
.Ltmp29399:
.LBB234_228:
	mov rcx, qword ptr [rsp + 8]
.Ltmp29400:
	mov eax, dword ptr [rcx]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [rcx], eax
.Ltmp29401:
	add r9d, 512
	and r9d, -1342177792
	mov rax, qword ptr [rsp + 64]
	mov dword ptr [rax], r9d
	lea rax, [rsp + 22]
.Ltmp29403:
	mov qword ptr [rsp + 144], rax
	lea rcx, [rip + <bool as core::fmt::Display>::fmt]
	mov qword ptr [rsp + 152], rcx
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.166]
	mov eax, 2
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.164]
.Ltmp29404:
	lea r8, [rsp + 144]
	mov edi, 24
	lea r10, [rsp + 23]
	lea r9, [rsp + 160]
	jmp .LBB234_231
.Ltmp29405:
.LBB234_229:
	mov rbx, qword ptr [rsp + 64]
	mov r15d, r9d
	mov r12, qword ptr [rsp + 8]
.Ltmp29406:
.LBB234_230:
	mov eax, dword ptr [r12]
	add eax, 512
	and eax, -1342177792
	#MEMBARRIER
	mov dword ptr [r12], eax
.Ltmp29407:
	add r15d, 512
.Ltmp29408:
	and r15d, -1342177792
	mov dword ptr [rbx], r15d
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.171]
	mov eax, 1
	lea rdx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.170]
	lea rcx, [rip + core::fmt::num::imp::<impl core::fmt::Display for u64>::fmt]
	lea r9, [rsp + 144]
	mov edi, 8
	lea r10, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.167]
	mov r8, r9
.Ltmp29409:
.LBB234_231:
	mov qword ptr [r9], r10
	mov qword ptr [r8 + rdi], rcx
	mov rdi, qword ptr [rsp + 184]
	mov qword ptr [rdi], rdx
	mov qword ptr [rdi + 8], 2
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], r8
	mov qword ptr [rdi + 24], rax
	call core::panicking::panic_fmt
.Ltmp29410:
.LBB234_232:
	dec rax
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.205]
.Ltmp29411:
.LBB234_233:
	mov qword ptr [rsp + 96], rcx
	mov r14, rax
.LBB234_234:
.Ltmp28598:
	mov esi, 15
	mov rdi, r14
	mov rdx, qword ptr [rsp + 96]
	call core::panicking::panic_bounds_check
.Ltmp28599:
	jmp .LBB234_243
.Ltmp29413:
.LBB234_235:
	mov r14d, 15
.Ltmp29414:
	jmp .LBB234_234
.Ltmp29415:
.LBB234_236:
	dec rcx
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.205]
.Ltmp29416:
	mov qword ptr [rsp + 96], rax
	mov r15d, esi
.Ltmp29417:
	mov r14, rcx
.Ltmp29418:
	jmp .LBB234_234
.Ltmp29419:
.LBB234_237:
	mov qword ptr [rsp + 96], r8
	mov r14d, 15
.Ltmp29420:
	mov r15d, esi
.Ltmp29421:
	jmp .LBB234_234
.Ltmp29422:
.LBB234_238:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.204]
	mov qword ptr [rsp + 176], rax
.LBB234_239:
	mov rax, qword ptr [rsp + 176]
	mov qword ptr [rsp + 96], rax
	jmp .LBB234_234
.Ltmp29424:
.LBB234_240:
	mov qword ptr [rsp + 136], rax
	lea rax, [rsp + 128]
.Ltmp29426:
	mov qword ptr [rsp + 144], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 152], rax
	lea rcx, [rsp + 136]
	mov qword ptr [rsp + 160], rcx
	mov qword ptr [rsp + 168], rax
.Ltmp29427:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.175]
.Ltmp29428:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.177]
	jmp .LBB234_242
.Ltmp29429:
.LBB234_241:
	mov qword ptr [rsp + 136], rax
	lea rax, [rsp + 128]
.Ltmp29431:
	mov qword ptr [rsp + 144], rax
	lea rax, [rip + <*mut T as core::fmt::Debug>::fmt]
	mov qword ptr [rsp + 152], rax
	lea rcx, [rsp + 136]
	mov qword ptr [rsp + 160], rcx
	mov qword ptr [rsp + 168], rax
.Ltmp29432:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.180]
.Ltmp29433:
	mov qword ptr [rsp + 192], rax
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.181]
.Ltmp29434:
.LBB234_242:
	lea rax, [rsp + 144]
	lea rdi, [rsp + 192]
	mov qword ptr [rdi + 8], 3
	mov qword ptr [rdi + 32], 0
	mov qword ptr [rdi + 16], rax
	mov qword ptr [rdi + 24], 2
.Ltmp28623:
	call core::panicking::panic_fmt
.Ltmp29435:
.Ltmp28624:
.LBB234_243:
	ud2
.Ltmp29436:
.LBB234_244:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.204]
	mov qword ptr [rsp + 96], rax
	mov r15d, esi
.Ltmp29437:
	jmp .LBB234_234
.Ltmp29438:
.LBB234_245:
	mov r14d, 15
	jmp .LBB234_239
.Ltmp29439:
.Ltmp28586:
	mov r14, rax
	jmp .LBB234_282
.Ltmp29440:
.Ltmp28597:
	jmp .LBB234_274
.Ltmp29441:
.Ltmp28607:
	mov r12d, dword ptr [rsp + 24]
.Ltmp29442:
	mov r14, rax
	xor ecx, ecx
.Ltmp29443:
	mov al, 1
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29444:
	je .LBB234_279
.Ltmp28608:
	mov rdi, qword ptr [rsp]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28609:
	jmp .LBB234_279
.Ltmp29446:
.Ltmp28610:
	call core::panicking::panic_in_cleanup
.Ltmp29447:
.Ltmp28617:
	mov r12d, dword ptr [rsp + 24]
.Ltmp29448:
	mov r14, rax
	xor ecx, ecx
.Ltmp29449:
	mov al, 1
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29450:
	je .LBB234_279
.Ltmp28618:
	mov rdi, qword ptr [rsp]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28619:
	jmp .LBB234_279
.Ltmp29452:
.Ltmp28620:
	call core::panicking::panic_in_cleanup
.Ltmp29453:
.Ltmp28629:
	mov r12d, dword ptr [rsp + 24]
.Ltmp29454:
	mov r14, rax
	xor ecx, ecx
.Ltmp29455:
	mov al, 1
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29456:
	je .LBB234_279
.Ltmp28630:
	mov rdi, qword ptr [rsp]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28631:
	jmp .LBB234_279
.Ltmp29458:
.Ltmp28632:
	call core::panicking::panic_in_cleanup
.Ltmp29459:
.Ltmp28639:
	mov r12d, dword ptr [rsp + 24]
.Ltmp29460:
	mov r14, rax
	xor ecx, ecx
.Ltmp29461:
	mov al, 1
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29462:
	je .LBB234_279
.Ltmp28640:
	mov rdi, qword ptr [rsp]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28641:
	jmp .LBB234_279
.Ltmp29464:
.Ltmp28642:
	call core::panicking::panic_in_cleanup
.Ltmp29465:
.Ltmp28580:
	mov r14, rax
	xor ecx, ecx
.Ltmp29466:
	mov al, 1
	lock cmpxchg	byte ptr [r12], cl
.Ltmp29467:
	je .LBB234_282
.Ltmp28581:
	mov rdi, r12
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28582:
	jmp .LBB234_282
.Ltmp29469:
.Ltmp28583:
	call core::panicking::panic_in_cleanup
.Ltmp29470:
.Ltmp28591:
	mov r14, rax
	xor ecx, ecx
.Ltmp29471:
	mov al, 1
	mov rdx, qword ptr [rsp]
	lock cmpxchg	byte ptr [rdx], cl
.Ltmp29472:
	je .LBB234_275
.Ltmp28592:
	mov rdi, qword ptr [rsp]
	call parking_lot::raw_mutex::RawMutex::unlock_slow
.Ltmp28593:
	mov r12d, dword ptr [rsp + 24]
	jmp .LBB234_276
.Ltmp29474:
.Ltmp28594:
	call core::panicking::panic_in_cleanup
.Ltmp29475:
.Ltmp28568:
	mov r14, rax
	mov r12, qword ptr [rsp + 56]
	jmp .LBB234_281
.Ltmp29476:
.Ltmp28559:
	mov r14, rax
.Ltmp29477:
	jmp .LBB234_271
.Ltmp29478:
.Ltmp28647:
	mov r14, rax
	mov r12d, dword ptr [rsp + 24]
	jmp .LBB234_279
.Ltmp29479:
.Ltmp28650:
	mov r14, rax
	lea rdi, [rsp + 192]
.Ltmp29480:
	call core::ptr::drop_in_place<masstree::leaf24::LeafNode24<masstree::value::LeafValueIndex<u64>>>
.Ltmp29481:
.LBB234_271:
	test r15b, 4
	jne .LBB234_282
	mov eax, r15d
	and eax, 2
	lea eax, [r15 + 4*rax]
	and eax, -268435464
	mov dword ptr [rbx], eax
	mov rdi, r14
	call _Unwind_Resume@PLT
.Ltmp29483:
.Ltmp28602:
.LBB234_274:
	mov r14, rax
.LBB234_275:
	mov r12d, dword ptr [rsp + 24]
.Ltmp29485:
.LBB234_276:
	test r15b, 4
	jne .LBB234_278
	mov eax, r15d
	and eax, 2
	lea eax, [r15 + 4*rax]
	and eax, -268435464
	mov dword ptr [rbx], eax
.Ltmp29487:
	jmp .LBB234_279
.Ltmp29488:
.LBB234_278:
	add r15d, 512
	and r15d, -1342177792
	mov dword ptr [rbx], r15d
.Ltmp29489:
.LBB234_279:
	add r12d, 512
	and r12d, -1342177792
	mov rax, qword ptr [rsp + 64]
	mov dword ptr [rax], r12d
	mov rdi, r14
	call _Unwind_Resume@PLT
.Ltmp29490:
.Ltmp28575:
	mov r14, rax
.Ltmp29491:
.LBB234_281:
	mov rdi, r12
	call core::ptr::drop_in_place<masstree::leaf24::LeafNode24<masstree::value::LeafValueIndex<u64>>>
.Ltmp29492:
	mov rdi, r12
	call qword ptr [rip + mi_free@GOTPCREL]
.Ltmp29493:
.LBB234_282:
	add r15d, 512
.Ltmp29494:
	and r15d, -1342177792
	mov dword ptr [rbx], r15d
	mov rdi, r14
	call _Unwind_Resume@PLT
.Ltmp29495:
.Lfunc_end234:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic, .Lfunc_end234-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic","a",@progbits
	.p2align	2, 0x0
