masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert:
.Lfunc_begin157:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception51
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
	sub rsp, 248
	.cfi_def_cfa_offset 304
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	.cfi_offset rbp, -16
	mov rbx, r8
	mov qword ptr [rsp], rsi
.Ltmp15979:
	mov qword ptr [rsp + 16], rdi
.Ltmp15980:
	test byte ptr fs:[seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF], 1
	mov qword ptr [rsp + 104], rcx
.Ltmp15981:
	je .LBB157_236
.Ltmp15982:
	mov rax, qword ptr fs:[0]
	lea rax, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
.Ltmp15983:
	mov rsi, qword ptr [rax + 24]
	mov qword ptr [rsp + 48], rsi
	movups xmm0, xmmword ptr [rax + 8]
	movaps xmmword ptr [rsp + 32], xmm0
.Ltmp15984:
.LBB157_2:
	mov r12, qword ptr [rsp + 40]
	movaps xmm0, xmmword ptr [rsp + 32]
	mov rsi, qword ptr [rsp + 48]
	mov rax, qword ptr [rsp]
	mov qword ptr [rsp + 88], rsi
.Ltmp15987:
	mov r15, qword ptr [rax + 8*rsi + 472]
.Ltmp15988:
	test r15, r15
	je .LBB157_237
.Ltmp15989:
.LBB157_3:
	shl r12, 8
.Ltmp15990:
	movzx eax, byte ptr [r15 + r12 + 128]
.Ltmp15991:
	add r15, r12
.Ltmp15992:
	test al, al
	je .LBB157_238
.Ltmp15993:
.LBB157_4:
	mov rax, qword ptr [r15 + 8]
.Ltmp15994:
	lea rsi, [rax + 1]
.Ltmp15995:
	mov qword ptr [r15 + 8], rsi
.Ltmp15996:
	test rax, rax
	jne .LBB157_9
.Ltmp15997:
	movzx eax, byte ptr [rip + seize::raw::membarrier::linux::STRATEGY.0]
.Ltmp15998:
	cmp al, 2
	jne .LBB157_7
.Ltmp15999:
	xor eax, eax
.Ltmp16000:
	xchg qword ptr [r15], rax
.Ltmp16001:
	jmp .LBB157_8
.Ltmp16002:
.LBB157_7:
	mov qword ptr [r15], 0
.Ltmp16003:
.LBB157_8:
	#MEMBARRIER
.LBB157_9:
	mov rax, qword ptr [rsp]
	mov qword ptr [rsp + 176], rax
	movups xmmword ptr [rsp + 184], xmm0
	mov rax, qword ptr [rsp + 88]
	mov qword ptr [rsp + 200], rax
	mov qword ptr [rsp + 168], r15
.Ltmp16004:
	mov qword ptr [rsp + 208], r15
.Ltmp16005:
	cmp rcx, 257
	jae .LBB157_239
.Ltmp16006:
	cmp rcx, 8
.Ltmp16007:
	jae .LBB157_13
.Ltmp16008:
	test rcx, rcx
.Ltmp16009:
	jne .LBB157_241
.Ltmp16010:
	xor eax, eax
	jmp .LBB157_14
.LBB157_13:
	mov rax, qword ptr [rdx]
.Ltmp16012:
	bswap rax
.Ltmp16013:
.LBB157_14:
	cmp rcx, 8
	mov esi, 8
	cmovb rsi, rcx
.Ltmp16014:
	mov qword ptr [rsp + 32], rdx
	mov qword ptr [rsp + 40], rcx
	mov qword ptr [rsp + 48], rax
	mov qword ptr [rsp + 56], 0
	mov qword ptr [rsp + 64], rsi
.Ltmp16015:
	mov edi, 24
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp16017:
	test rax, rax
	je .LBB157_240
.Ltmp16018:
	mov qword ptr [rsp + 80], r12
	mov qword ptr [rax], 1
	mov qword ptr [rax + 8], 1
	mov rcx, rax
	add rcx, 16
	mov qword ptr [rsp + 136], rcx
	mov qword ptr [rax + 16], rbx
	mov qword ptr [rsp + 96], rax
.Ltmp16019:
	mov qword ptr [rsp + 152], rax
.Ltmp16020:
	mov rax, qword ptr [rsp]
.Ltmp16021:
	mov rbx, qword ptr [rax + 1024]
.Ltmp16022:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
	mov qword ptr [rsp + 160], rax
.Ltmp16023:
	jmp .LBB157_17
.Ltmp16024:
	.p2align	4
.LBB157_16:
	mov rax, qword ptr [rsp]
.Ltmp16026:
	mov rbx, qword ptr [rax + 1024]
.Ltmp16027:
.LBB157_17:
	mov r12, rbx
	lea r14, [rbx + 264]
	lea rcx, [rbx + 560]
	mov qword ptr [rsp + 24], r14
	mov qword ptr [rsp + 8], rcx
	jmp .LBB157_20
.Ltmp16028:
.LBB157_18:
	mov rdi, r13
	call masstree::leaf24::LeafNode24<S>::wait_for_split
.Ltmp16029:
	.p2align	4
.LBB157_19:
	lea eax, [rbp + 512]
	mov ecx, ebp
	add ecx, 8
	test bpl, 4
	mov edx, -1342177792
	mov esi, -268435464
	cmove edx, esi
	cmovne ecx, eax
	and ecx, edx
	mov dword ptr [r13], ecx
	mov r14, qword ptr [rsp + 24]
	mov rcx, qword ptr [rsp + 8]
.Ltmp16031:
.LBB157_20:
	mov eax, dword ptr [r12]
.Ltmp16032:
	test eax, eax
.Ltmp16033:
	mov rax, r14
	cmovs rax, rcx
	mov rbx, qword ptr [rax]
.Ltmp16034:
	test rbx, rbx
.Ltmp16035:
	jne .LBB157_17
.Ltmp16036:
	mov rbx, qword ptr [rsp + 48]
	mov rcx, r12
.Ltmp16037:
	.p2align	4
.LBB157_22:
	mov r13, rcx
.Ltmp16039:
	mov ebp, dword ptr [rcx]
	test bpl, 6
	je .LBB157_28
	xor eax, eax
	jmp .LBB157_25
	.p2align	4
.LBB157_24:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ebp, dword ptr [r13]
	test bpl, 6
	je .LBB157_28
.LBB157_25:
	xor ecx, ecx
	.p2align	4
.LBB157_26:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB157_24
	cmp ecx, eax
	jbe .LBB157_26
	jmp .LBB157_24
.Ltmp16045:
	.p2align	4
.LBB157_28:
	#MEMBARRIER
	mov eax, dword ptr [r13]
.Ltmp16047:
	test eax, eax
.Ltmp16048:
	js .LBB157_35
.Ltmp16049:
.Ltmp15902:
	mov rdi, rbx
	mov rsi, r13
	call masstree::ksearch::upper_bound_internode_generic
.Ltmp15903:
.Ltmp16050:
	cmp eax, 15
	jne .LBB157_32
.Ltmp16051:
	mov rax, qword ptr [r13 + 256]
.Ltmp16052:
	mov rcx, r12
.Ltmp16053:
	test rax, rax
.Ltmp16054:
	jne .LBB157_33
	jmp .LBB157_22
.Ltmp16055:
	.p2align	4
.LBB157_32:
	mov rax, qword ptr [r13 + 8*rax + 136]
.Ltmp16057:
	mov rcx, r12
.Ltmp16058:
	test rax, rax
.Ltmp16059:
	je .LBB157_22
.Ltmp16060:
.LBB157_33:
	prefetcht0 byte ptr [rax]
.Ltmp16061:
	#MEMBARRIER
	mov edx, dword ptr [r13]
.Ltmp16062:
	xor edx, ebp
	mov rcx, rax
	cmp edx, 4
.Ltmp16063:
	jb .LBB157_22
.Ltmp16064:
	#MEMBARRIER
	mov eax, dword ptr [r13]
.Ltmp16065:
	xor eax, ebp
	cmp eax, 512
.Ltmp16066:
	cmovae r13, r12
.Ltmp16067:
	mov rcx, r13
.Ltmp16068:
	jmp .LBB157_22
.Ltmp16069:
	.p2align	4
.LBB157_35:
	mov rax, qword ptr [rsp + 48]
.Ltmp16071:
	mov ecx, dword ptr [r13]
.Ltmp16072:
	test cl, 4
.Ltmp16073:
	jne .LBB157_37
	xor ecx, ecx
.Ltmp16075:
	jmp .LBB157_44
.Ltmp16076:
	.p2align	4
.LBB157_37:
	mov ecx, dword ptr [r13]
	test cl, 6
	je .LBB157_43
	xor ecx, ecx
	jmp .LBB157_40
	.p2align	4
.LBB157_39:
	and ecx, 7
	lea ecx, [2*rcx + 1]
	mov edx, dword ptr [r13]
	test dl, 6
	je .LBB157_43
.LBB157_40:
	xor edx, edx
	.p2align	4
.LBB157_41:
	mov esi, edx
	pause
	cmp edx, ecx
	adc edx, 0
	cmp esi, ecx
	jae .LBB157_39
	cmp edx, ecx
	jbe .LBB157_41
	jmp .LBB157_39
	.p2align	4
.LBB157_43:
	#MEMBARRIER
	xor ecx, ecx
.Ltmp16085:
	.p2align	4
.LBB157_44:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16086:
	test sil, 1
.Ltmp16087:
	jne .LBB157_48
.Ltmp16088:
.LBB157_45:
	test rsi, rsi
.Ltmp16089:
	je .LBB157_74
.Ltmp16090:
	mov rdx, qword ptr [rsi + 128]
.Ltmp16091:
	cmp rax, rdx
.Ltmp16092:
	jb .LBB157_74
	inc rcx
.Ltmp16094:
	mov r13, rsi
.Ltmp16095:
	cmp rcx, 128
.Ltmp16096:
	jne .LBB157_44
	jmp .LBB157_16
	.p2align	4
.LBB157_48:
	xor edx, edx
	jmp .LBB157_50
.Ltmp16099:
	.p2align	4
.LBB157_49:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16100:
	mov edx, 0
.Ltmp16101:
	test sil, 1
	je .LBB157_45
.Ltmp16102:
.LBB157_50:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16103:
	test sil, 1
.Ltmp16104:
	je .LBB157_49
.Ltmp16105:
	pause
.Ltmp16106:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16107:
	test sil, 1
.Ltmp16108:
	je .LBB157_49
.Ltmp16109:
	pause
.Ltmp16110:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16111:
	test sil, 1
.Ltmp16112:
	je .LBB157_49
.Ltmp16113:
	pause
.Ltmp16114:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16115:
	test sil, 1
.Ltmp16116:
	je .LBB157_49
.Ltmp16117:
	pause
.Ltmp16118:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16119:
	test sil, 1
.Ltmp16120:
	je .LBB157_49
.Ltmp16121:
	pause
.Ltmp16122:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16123:
	test sil, 1
.Ltmp16124:
	je .LBB157_49
.Ltmp16125:
	pause
.Ltmp16126:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16127:
	test sil, 1
.Ltmp16128:
	je .LBB157_49
.Ltmp16129:
	pause
.Ltmp16130:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16131:
	test sil, 1
.Ltmp16132:
	je .LBB157_49
.Ltmp16133:
	pause
.Ltmp16134:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16135:
	test sil, 1
.Ltmp16136:
	je .LBB157_49
.Ltmp16137:
	pause
.Ltmp16138:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16139:
	test sil, 1
.Ltmp16140:
	je .LBB157_49
.Ltmp16141:
	pause
.Ltmp16142:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16143:
	test sil, 1
.Ltmp16144:
	je .LBB157_49
.Ltmp16145:
	pause
.Ltmp16146:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16147:
	test sil, 1
.Ltmp16148:
	je .LBB157_49
.Ltmp16149:
	pause
.Ltmp16150:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16151:
	test sil, 1
.Ltmp16152:
	je .LBB157_49
.Ltmp16153:
	pause
.Ltmp16154:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16155:
	test sil, 1
.Ltmp16156:
	je .LBB157_49
.Ltmp16157:
	pause
.Ltmp16158:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16159:
	test sil, 1
.Ltmp16160:
	je .LBB157_49
.Ltmp16161:
	pause
.Ltmp16162:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16163:
	test sil, 1
.Ltmp16164:
	je .LBB157_49
.Ltmp16165:
	pause
.Ltmp16166:
	mov rsi, qword ptr [r13 + 544]
.Ltmp16167:
	test sil, 1
.Ltmp16168:
	je .LBB157_49
.Ltmp16169:
	mov esi, dword ptr [r13]
	test sil, 6
	je .LBB157_73
.Ltmp16170:
	xor esi, esi
	jmp .LBB157_70
	.p2align	4
.LBB157_69:
	and esi, 7
	lea esi, [2*rsi + 1]
	mov edi, dword ptr [r13]
	test dil, 6
	je .LBB157_73
.LBB157_70:
	xor edi, edi
	.p2align	4
.LBB157_71:
	mov r8d, edi
	pause
	cmp edi, esi
	adc edi, 0
	cmp r8d, esi
	jae .LBB157_69
	cmp edi, esi
	jbe .LBB157_71
	jmp .LBB157_69
.LBB157_73:
	#MEMBARRIER
	inc rdx
	cmp rdx, 1001
	jne .LBB157_50
	jmp .LBB157_49
.Ltmp16177:
	.p2align	4
.LBB157_74:
	mov ebx, dword ptr [r13]
	test bl, 6
	je .LBB157_80
	xor eax, eax
	jmp .LBB157_77
	.p2align	4
.LBB157_76:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ebx, dword ptr [r13]
	test bl, 6
	je .LBB157_80
.LBB157_77:
	xor ecx, ecx
	.p2align	4
.LBB157_78:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB157_76
	cmp ecx, eax
	jbe .LBB157_78
	jmp .LBB157_76
.Ltmp16184:
	.p2align	4
.LBB157_80:
	#MEMBARRIER
	lea rdi, [r13 + 64]
.Ltmp16186:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp15905:
	mov qword ptr [rsp + 112], rdi
.Ltmp16188:
	call rax
.Ltmp16189:
.Ltmp15906:
	mov r14, rax
	xor ecx, ecx
	jmp .LBB157_83
	.p2align	4
.LBB157_82:
	and ecx, 7
	lea ecx, [2*rcx + 1]
.LBB157_83:
	mov ebp, dword ptr [r13]
	test bpl, 1
	jne .LBB157_85
	mov r15d, ebp
	or r15d, 3
	mov eax, ebp
	lock cmpxchg	dword ptr [r13], r15d
	je .LBB157_88
.LBB157_85:
	xor eax, eax
	.p2align	4
.LBB157_86:
	mov esi, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp esi, ecx
	jae .LBB157_82
	cmp eax, ecx
	jbe .LBB157_86
	jmp .LBB157_82
.Ltmp16196:
	.p2align	4
.LBB157_88:
	#MEMBARRIER
	mov eax, dword ptr [r13]
.Ltmp16198:
	xor eax, ebx
	cmp eax, 3
.Ltmp16199:
	ja .LBB157_19
	mov rbx, rdx
.Ltmp16201:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp15908:
	mov rdi, qword ptr [rsp + 112]
.Ltmp16203:
	call rax
.Ltmp16204:
	xor rax, r14
	xor rdx, rbx
	or rdx, rax
.Ltmp16205:
	jne .LBB157_19
.Ltmp16206:
	mov rax, qword ptr [r13 + 544]
.Ltmp16207:
	test al, 1
	jne .LBB157_18
.Ltmp16208:
	test rax, rax
.Ltmp16209:
	je .LBB157_94
.Ltmp16210:
	mov rax, qword ptr [rax + 128]
.Ltmp16211:
	cmp qword ptr [rsp + 48], rax
	jae .LBB157_19
.Ltmp16212:
.LBB157_94:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
	mov rdi, qword ptr [rsp + 112]
.Ltmp16214:
	call rax
.Ltmp16215:
.Ltmp15911:
	mov r10, rax
	mov r11, rdx
.Ltmp16216:
	cmp qword ptr [rsp + 104], 9
.Ltmp16217:
	mov qword ptr [rsp + 144], rax
.Ltmp16218:
	jb .LBB157_115
.Ltmp16219:
	mov rsi, qword ptr [rsp + 40]
.Ltmp16220:
	mov r8, qword ptr [rsp + 48]
.Ltmp16221:
	mov rax, qword ptr [rsp + 56]
	shl rax, 3
.Ltmp16222:
	mov rdi, rsi
	sub rdi, rax
	mov eax, 0
.Ltmp16223:
	cmovb rdi, rax
.Ltmp16224:
	cmp rdi, 9
.Ltmp16225:
	mov edx, edi
.Ltmp16226:
	mov eax, 64
	cmovae edx, eax
.Ltmp16227:
	mov rax, r10
	and rax, 31
.Ltmp16228:
	je .LBB157_126
.Ltmp16229:
	mov ecx, 5
	xor ebx, ebx
.Ltmp16230:
	cmp rdi, 8
	ja .LBB157_100
	jmp .LBB157_108
.Ltmp16231:
	.p2align	4
.LBB157_98:
	cmp rdi, r8
	ja .LBB157_114
.Ltmp16233:
.LBB157_99:
	inc rbx
.Ltmp16234:
	add rcx, 5
	cmp rax, rbx
.Ltmp16235:
	je .LBB157_137
.Ltmp16236:
.LBB157_100:
	mov rdi, r10
	shrd rdi, r11, cl
	mov r14, r11
	shr r14, cl
	test cl, 64
	cmove r14, rdi
	and r14d, 31
.Ltmp16237:
	cmp r14, 23
	ja .LBB157_255
.Ltmp16238:
	mov rdi, qword ptr [r13 + 8*r14 + 128]
.Ltmp16239:
	cmp rdi, r8
	jne .LBB157_98
.Ltmp16240:
	movzx r9d, byte ptr [r13 + r14 + 320]
.Ltmp16241:
	mov r10, qword ptr [r13 + 8*r14 + 344]
.Ltmp16242:
	test r10, r10
	mov r10, qword ptr [rsp + 144]
.Ltmp16243:
	je .LBB157_99
.Ltmp16244:
	test r9b, r9b
	js .LBB157_156
	cmp r9b, dl
	je .LBB157_148
	cmp r9b, 64
.Ltmp16247:
	jne .LBB157_98
	jmp .LBB157_187
.Ltmp16248:
	.p2align	4
.LBB157_106:
	ja .LBB157_114
.LBB157_107:
	inc rbx
.Ltmp16251:
	add rcx, 5
	cmp rax, rbx
.Ltmp16252:
	je .LBB157_137
.Ltmp16253:
.LBB157_108:
	mov r9, r10
	shrd r9, r11, cl
	mov r14, r11
	shr r14, cl
	test cl, 64
	cmove r14, r9
	and r14d, 31
.Ltmp16254:
	cmp r14, 23
	ja .LBB157_255
.Ltmp16255:
	mov r9, qword ptr [r13 + 8*r14 + 128]
.Ltmp16256:
	cmp r9, r8
	jne .LBB157_106
.Ltmp16257:
	movzx r9d, byte ptr [r13 + r14 + 320]
.Ltmp16258:
	mov r10, qword ptr [r13 + 8*r14 + 344]
.Ltmp16259:
	test r10, r10
	mov r10, qword ptr [rsp + 144]
.Ltmp16260:
	je .LBB157_107
.Ltmp16261:
	movzx r9d, r9b
.Ltmp16262:
	test r9b, r9b
	js .LBB157_114
	cmp r9b, dl
	je .LBB157_148
.Ltmp16264:
	cmp rdi, r9
.Ltmp16265:
	jae .LBB157_107
.Ltmp16266:
.LBB157_114:
	mov r8, qword ptr [rsp + 48]
.Ltmp16267:
	cmp eax, 23
	ja .LBB157_138
	jmp .LBB157_139
.Ltmp16268:
.LBB157_115:
	mov rdx, qword ptr [rsp + 40]
.Ltmp16269:
	mov r8, qword ptr [rsp + 48]
.Ltmp16270:
	mov rax, qword ptr [rsp + 56]
	shl rax, 3
.Ltmp16271:
	sub rdx, rax
.Ltmp16272:
	mov eax, 0
.Ltmp16273:
	cmovb rdx, rax
.Ltmp16274:
	mov rax, r10
	and rax, 31
.Ltmp16275:
	je .LBB157_127
.Ltmp16276:
	mov ecx, 5
	xor ebx, ebx
	jmp .LBB157_119
.Ltmp16277:
	.p2align	4
.LBB157_117:
	ja .LBB157_125
.Ltmp16278:
.LBB157_118:
	inc rbx
.Ltmp16279:
	add rcx, 5
	cmp rax, rbx
.Ltmp16280:
	je .LBB157_128
.Ltmp16281:
.LBB157_119:
	mov rsi, r10
	shrd rsi, r11, cl
	mov r14, r11
	shr r14, cl
	test cl, 64
	cmove r14, rsi
	and r14d, 31
.Ltmp16282:
	cmp r14, 24
	jae .LBB157_255
.Ltmp16283:
	mov rsi, qword ptr [r13 + 8*r14 + 128]
.Ltmp16284:
	cmp rsi, r8
	jne .LBB157_117
.Ltmp16285:
	movzx esi, byte ptr [r13 + r14 + 320]
.Ltmp16286:
	mov rdi, qword ptr [r13 + 8*r14 + 344]
.Ltmp16287:
	test rdi, rdi
.Ltmp16288:
	je .LBB157_118
	test sil, sil
	js .LBB157_125
	cmp sil, dl
	jne .LBB157_117
.Ltmp16291:
	mov r15, qword ptr [r13 + 8*r14 + 344]
.Ltmp16292:
	test r15, r15
.Ltmp16293:
	je .LBB157_19
	jmp .LBB157_203
.Ltmp16294:
.LBB157_125:
	mov r8, qword ptr [rsp + 48]
.Ltmp16295:
	cmp eax, 23
	ja .LBB157_129
	jmp .LBB157_130
.Ltmp16296:
.LBB157_126:
	xor ebx, ebx
	jmp .LBB157_139
.LBB157_127:
	xor ebx, ebx
	jmp .LBB157_130
.LBB157_128:
	mov rbx, rax
.Ltmp16299:
	mov r8, qword ptr [rsp + 48]
.Ltmp16300:
	cmp eax, 23
	jbe .LBB157_130
.LBB157_129:
.Ltmp15922:
	mov rdi, qword ptr [rsp]
.Ltmp16302:
	mov rsi, r13
	mov rdx, r13
	mov ecx, r15d
	lea r9, [rsp + 176]
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
.Ltmp16303:
	mov r14, qword ptr [rsp + 24]
	jmp .LBB157_146
.Ltmp16304:
.LBB157_130:
	mov r9, r11
	shr r9, 56
.Ltmp16305:
	mov r14, r9
	and r14, 31
.Ltmp16306:
	jne .LBB157_198
.Ltmp16307:
	mov rcx, qword ptr [r13 + 552]
.Ltmp16308:
	test rcx, rcx
.Ltmp16309:
	je .LBB157_195
.Ltmp16310:
	mov rcx, qword ptr [r13 + 128]
.Ltmp16311:
	cmp rcx, r8
.Ltmp16312:
	je .LBB157_195
.Ltmp16313:
	add rax, -23
.Ltmp16314:
	mov ecx, 115
	xor edx, edx
.Ltmp16315:
	.p2align	4
.LBB157_134:
	cmp rax, rdx
.Ltmp16316:
	je .LBB157_136
.Ltmp16317:
	mov rsi, r10
	shrd rsi, r11, cl
	mov r14, r11
	shr r14, cl
	test cl, 64
	cmove r14, rsi
.Ltmp16318:
	dec rdx
.Ltmp16319:
	add rcx, -5
	and r14, 31
.Ltmp16320:
	je .LBB157_134
	jmp .LBB157_164
.Ltmp16321:
.LBB157_136:
	mov rdi, qword ptr [rsp]
.Ltmp16322:
	mov rsi, r13
	mov rdx, r13
	mov ecx, r15d
	lea r9, [rsp + 176]
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
.Ltmp16323:
	mov r14, qword ptr [rsp + 24]
	jmp .LBB157_146
.Ltmp16324:
.LBB157_137:
	mov rbx, rax
.Ltmp16325:
	mov r8, qword ptr [rsp + 48]
.Ltmp16326:
	cmp eax, 23
	jbe .LBB157_139
.Ltmp16327:
.LBB157_138:
	mov rdi, qword ptr [rsp]
	mov rsi, r13
	mov rdx, r13
	mov ecx, r15d
	lea r9, [rsp + 176]
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
.Ltmp16329:
	mov r14, qword ptr [rsp + 24]
	jmp .LBB157_146
.Ltmp16330:
.LBB157_139:
	mov r9, r11
	shr r9, 56
.Ltmp16331:
	mov r14, r9
	and r14, 31
.Ltmp16332:
	jne .LBB157_199
.Ltmp16333:
	mov rcx, qword ptr [r13 + 552]
.Ltmp16334:
	test rcx, rcx
.Ltmp16335:
	je .LBB157_196
.Ltmp16336:
	mov rcx, qword ptr [r13 + 128]
.Ltmp16337:
	cmp rcx, r8
.Ltmp16338:
	je .LBB157_196
.Ltmp16339:
	add rax, -23
.Ltmp16340:
	mov ecx, 115
	xor edx, edx
.Ltmp16341:
	.p2align	4
.LBB157_143:
	cmp rax, rdx
.Ltmp16342:
	je .LBB157_145
.Ltmp16343:
	mov rsi, r10
	shrd rsi, r11, cl
	mov r14, r11
	shr r14, cl
	test cl, 64
	cmove r14, rsi
.Ltmp16344:
	dec rdx
.Ltmp16345:
	add rcx, -5
	and r14, 31
.Ltmp16346:
	je .LBB157_143
	jmp .LBB157_172
.Ltmp16347:
.LBB157_145:
	mov rdi, qword ptr [rsp]
	mov rsi, r13
	mov rdx, r13
	mov ecx, r15d
	lea r9, [rsp + 176]
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
.Ltmp16349:
.Ltmp15964:
	mov r14, qword ptr [rsp + 24]
.Ltmp16350:
.LBB157_146:
	cmp al, 6
	mov rcx, qword ptr [rsp + 8]
	je .LBB157_20
	jmp .LBB157_197
.Ltmp16351:
.LBB157_148:
	cmp dl, 64
	jne .LBB157_155
.Ltmp16352:
	mov rax, qword ptr [rsp + 64]
.Ltmp16353:
	mov rdi, qword ptr [rsp + 32]
	add rdi, rax
	sub rsi, rax
	mov edx, 0
.Ltmp16354:
	mov eax, 1
	cmovbe rdi, rax
.Ltmp16355:
	cmovae rdx, rsi
.Ltmp16356:
	movzx eax, byte ptr [r13 + r14 + 320]
.Ltmp16357:
	cmp al, 64
.Ltmp16358:
	jne .LBB157_187
.Ltmp16359:
	mov rcx, qword ptr [r13 + 536]
.Ltmp16360:
	test rcx, rcx
.Ltmp16361:
	je .LBB157_187
.Ltmp16362:
	mov esi, dword ptr [rcx + 8*r14 + 24]
	mov eax, 4294967295
	cmp rsi, rax
	je .LBB157_187
	movzx r9d, word ptr [rcx + 8*r14 + 28]
.Ltmp16365:
	lea rax, [r9 + rsi]
.Ltmp16366:
	mov r8, qword ptr [rcx + 16]
.Ltmp16367:
	cmp rax, r8
.Ltmp16368:
	ja .LBB157_244
.Ltmp16369:
	cmp rdx, r9
	jne .LBB157_187
.Ltmp16370:
	add rsi, qword ptr [rcx + 8]
.Ltmp16371:
	call qword ptr [rip + bcmp@GOTPCREL]
.Ltmp16372:
	test eax, eax
.Ltmp16373:
	jne .LBB157_187
.Ltmp16374:
.LBB157_155:
	mov r15, qword ptr [r13 + 8*r14 + 344]
.Ltmp16375:
	test r15, r15
.Ltmp16376:
	je .LBB157_19
	jmp .LBB157_212
.Ltmp16377:
.LBB157_156:
	mov rbx, qword ptr [r13 + 8*r14 + 344]
.Ltmp16378:
	mov eax, ebp
	add eax, 8
	test bpl, 4
	mov ecx, -1342177792
	mov edx, -268435464
	cmove ecx, edx
	lea edx, [rbp + 512]
	cmovne eax, edx
	and eax, ecx
	mov dword ptr [r13], eax
.Ltmp16379:
	mov r14, qword ptr [rsp + 40]
.Ltmp16380:
	mov r15, qword ptr [rsp + 56]
	lea rax, [r15 + 1]
	mov qword ptr [rsp + 56], rax
	lea rax, [8*r15 + 8]
.Ltmp16381:
	mov rsi, r14
	sub rsi, rax
.Ltmp16382:
	jb .LBB157_157
.Ltmp16383:
	mov rdi, qword ptr [rsp + 32]
	add rdi, rax
.Ltmp16384:
	cmp rsi, 8
.Ltmp16385:
	jae .LBB157_161
.Ltmp16386:
	cmp r14, rax
.Ltmp16387:
	jne .LBB157_163
.LBB157_157:
	xor eax, eax
.Ltmp16389:
	jmp .LBB157_162
.Ltmp16390:
.LBB157_161:
	mov rax, qword ptr [rdi]
.Ltmp16391:
	bswap rax
.Ltmp16392:
.LBB157_162:
	lea rcx, [8*r15 + 16]
.Ltmp16393:
	mov qword ptr [rsp + 48], rax
.Ltmp16394:
	cmp r14, rcx
	cmovb rcx, r14
.Ltmp16395:
	mov qword ptr [rsp + 64], rcx
.Ltmp16396:
	jmp .LBB157_17
.Ltmp16397:
.LBB157_163:
.Ltmp15958:
	call masstree::key::Key::read_ikey_slow
.Ltmp16398:
.Ltmp15959:
	jmp .LBB157_162
.Ltmp16399:
.LBB157_164:
	neg rdx
	mov r15, rdx
.Ltmp16400:
.LBB157_165:
	mov rax, qword ptr [rsp + 48]
.Ltmp16401:
	mov rcx, qword ptr [rsp + 96]
.Ltmp16402:
	lock inc	qword ptr [rcx]
.Ltmp16403:
	jle .LBB157_256
.Ltmp16404:
	cmp r14d, 23
	ja .LBB157_254
.Ltmp16405:
	mov qword ptr [r13 + 8*r14 + 128], rax
	mov rax, qword ptr [rsp + 136]
.Ltmp16407:
	mov qword ptr [r13 + 8*r14 + 344], rax
.Ltmp16408:
	mov rax, qword ptr [rsp + 40]
	mov rdx, qword ptr [rsp + 56]
	shl rdx, 3
	xor ecx, ecx
.Ltmp16410:
	mov rsi, rax
	sub rsi, rdx
	cmovae rcx, rsi
.Ltmp16411:
	cmp rcx, 8
.Ltmp16412:
	jbe .LBB157_185
.Ltmp16413:
	mov byte ptr [r13 + r14 + 320], 64
.Ltmp16414:
	mov rdx, qword ptr [rsp + 64]
	mov rsi, qword ptr [rsp + 32]
	add rsi, rdx
	xor ecx, ecx
	sub rax, rdx
	cmovae rcx, rax
.Ltmp16415:
	mov edx, 1
	cmova rdx, rsi
.Ltmp16416:
.Ltmp15926:
	lea r8, [rsp + 176]
.Ltmp16417:
	mov rdi, r13
	mov rsi, r14
	mov r14, r11
.Ltmp16418:
	mov r12, r9
.Ltmp16419:
	call masstree::leaf24::LeafNode24<S>::assign_ksuf
.Ltmp16420:
	mov r9, r12
	mov r10, qword ptr [rsp + 144]
	mov r11, r14
.Ltmp16421:
	test r15, r15
	je .LBB157_171
.Ltmp16422:
.LBB157_170:
	mov al, 23
.Ltmp16423:
	sub al, r15b
	movzx eax, al
	lea ecx, [rax + 4*rax]
	add cl, 5
	mov rax, r10
	shrd rax, r11, cl
	mov rdx, r11
	shr rdx, cl
	test cl, 64
	cmove rdx, rax
	xor r9d, edx
	and r9d, 31
.Ltmp16424:
	mov rax, r9
	shl rax, cl
	xor edx, edx
.Ltmp16425:
	test cl, 64
.Ltmp16426:
	mov rsi, rax
	cmovne rsi, rdx
	shld rdx, r9, cl
.Ltmp16427:
	test cl, 64
.Ltmp16428:
	cmovne rdx, rax
	shl r9, 56
.Ltmp16429:
	or r9, rdx
	xor r11, r9
.Ltmp16430:
	xor r10, rsi
.Ltmp16431:
	mov r9, r11
	shr r9, 56
.Ltmp16432:
.LBB157_171:
	mov r14, r10
.Ltmp16433:
	lea ecx, [rbx + 4*rbx]
	add cl, 5
.Ltmp16434:
	mov edi, 31
	shl rdi, cl
	mov r8d, 31
	xor esi, esi
.Ltmp16435:
	test cl, 64
.Ltmp16436:
	mov rax, rdi
	cmovne rax, rsi
	xor edx, edx
	shld rdx, r8, cl
.Ltmp16437:
	test cl, 64
.Ltmp16438:
	cmovne rdx, rdi
	and r9d, 31
.Ltmp16439:
	mov r10, r9
	shl r9, cl
.Ltmp16440:
	test cl, 64
.Ltmp16441:
	mov rdi, r9
	cmovne rdi, rsi
	xor r8d, r8d
	shld r8, r10, cl
.Ltmp16442:
	test cl, 64
.Ltmp16443:
	cmovne r8, r9
.Ltmp16444:
	mov r10, -1
	shl r10, cl
	test cl, 64
	cmove rsi, r10
	mov rcx, -1
	cmove r10, rcx
.Ltmp16445:
	mov rcx, r14
	add rcx, 1
	mov r9, r11
	adc r9, 0
	shld r11, r14, 5
.Ltmp16446:
	and r11, r10
.Ltmp16447:
	not r10
.Ltmp16448:
	shl r14, 5
.Ltmp16449:
	and r14, rsi
.Ltmp16450:
	not rsi
.Ltmp16451:
	and rcx, rsi
	or rcx, rdi
	and r9, r10
	or r9, r8
	not rdx
	not rax
	and r11, rdx
	or r11, r9
	and r14, rax
	or r14, rcx
.Ltmp16452:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC]
.Ltmp16453:
	mov rdi, qword ptr [rsp + 112]
.Ltmp16454:
	mov rsi, r14
.Ltmp16455:
	mov rdx, r11
	call rax
.Ltmp16456:
	jmp .LBB157_180
.Ltmp16457:
.LBB157_172:
	neg rdx
	mov r15, rdx
.Ltmp16458:
.LBB157_173:
	mov rax, qword ptr [rsp + 48]
.Ltmp16459:
	mov rcx, qword ptr [rsp + 96]
.Ltmp16460:
	lock inc	qword ptr [rcx]
.Ltmp16461:
	jle .LBB157_256
.Ltmp16462:
	cmp r14d, 23
	ja .LBB157_254
.Ltmp16463:
	mov qword ptr [r13 + 8*r14 + 128], rax
	mov rax, qword ptr [rsp + 136]
.Ltmp16465:
	mov qword ptr [r13 + 8*r14 + 344], rax
.Ltmp16466:
	mov rax, qword ptr [rsp + 40]
	mov rdx, qword ptr [rsp + 56]
	shl rdx, 3
	xor ecx, ecx
.Ltmp16468:
	mov rsi, rax
	sub rsi, rdx
	cmovae rcx, rsi
.Ltmp16469:
	cmp rcx, 8
.Ltmp16470:
	jbe .LBB157_186
.Ltmp16471:
	mov byte ptr [r13 + r14 + 320], 64
.Ltmp16472:
	mov rdx, qword ptr [rsp + 64]
	mov rsi, qword ptr [rsp + 32]
	add rsi, rdx
	xor ecx, ecx
	sub rax, rdx
	cmovae rcx, rax
.Ltmp16473:
	mov edx, 1
	cmova rdx, rsi
.Ltmp16474:
	lea r8, [rsp + 176]
.Ltmp16475:
	mov rdi, r13
	mov rsi, r14
	mov r14, r11
.Ltmp16476:
	mov r12, r9
.Ltmp16477:
	call masstree::leaf24::LeafNode24<S>::assign_ksuf
.Ltmp16478:
	mov r9, r12
	mov r10, qword ptr [rsp + 144]
	mov r11, r14
.Ltmp16479:
	test r15, r15
	je .LBB157_179
.Ltmp16480:
.LBB157_178:
	mov al, 23
.Ltmp16481:
	sub al, r15b
	movzx eax, al
	lea ecx, [rax + 4*rax]
	add cl, 5
	mov rax, r10
	shrd rax, r11, cl
	mov rdx, r11
	shr rdx, cl
	test cl, 64
	cmove rdx, rax
	xor r9d, edx
	and r9d, 31
.Ltmp16482:
	mov rax, r9
	shl rax, cl
	xor edx, edx
.Ltmp16483:
	test cl, 64
.Ltmp16484:
	mov rsi, rax
	cmovne rsi, rdx
	shld rdx, r9, cl
.Ltmp16485:
	test cl, 64
.Ltmp16486:
	cmovne rdx, rax
	shl r9, 56
.Ltmp16487:
	or r9, rdx
	xor r11, r9
.Ltmp16488:
	xor r10, rsi
.Ltmp16489:
	mov r9, r11
	shr r9, 56
.Ltmp16490:
.LBB157_179:
	mov r14, r10
.Ltmp16491:
	lea ecx, [rbx + 4*rbx]
	add cl, 5
.Ltmp16492:
	mov edi, 31
	shl rdi, cl
	mov r8d, 31
	xor esi, esi
.Ltmp16493:
	test cl, 64
.Ltmp16494:
	mov rax, rdi
	cmovne rax, rsi
	xor edx, edx
	shld rdx, r8, cl
.Ltmp16495:
	test cl, 64
.Ltmp16496:
	cmovne rdx, rdi
	and r9d, 31
.Ltmp16497:
	mov r10, r9
	shl r9, cl
.Ltmp16498:
	test cl, 64
.Ltmp16499:
	mov rdi, r9
	cmovne rdi, rsi
	xor r8d, r8d
	shld r8, r10, cl
.Ltmp16500:
	test cl, 64
.Ltmp16501:
	cmovne r8, r9
.Ltmp16502:
	mov r10, -1
	shl r10, cl
	test cl, 64
	cmove rsi, r10
	mov rcx, -1
	cmove r10, rcx
.Ltmp16503:
	mov rcx, r14
	add rcx, 1
	mov r9, r11
	adc r9, 0
	shld r11, r14, 5
.Ltmp16504:
	and r11, r10
.Ltmp16505:
	not r10
.Ltmp16506:
	shl r14, 5
.Ltmp16507:
	and r14, rsi
.Ltmp16508:
	not rsi
.Ltmp16509:
	and rcx, rsi
	or rcx, rdi
	and r9, r10
	or r9, r8
	not rdx
	not rax
	and r11, rdx
	or r11, r9
	and r14, rax
	or r14, rcx
.Ltmp16510:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC]
	mov rdi, qword ptr [rsp + 112]
.Ltmp16512:
	mov rsi, r14
.Ltmp16513:
	mov rdx, r11
	call rax
.Ltmp16514:
.LBB157_180:
	mov eax, ebp
	add eax, 8
	test bpl, 4
	mov ecx, -268435464
	mov edx, -1342177792
	cmove edx, ecx
	lea ecx, [rbp + 512]
	cmovne eax, ecx
	and eax, edx
	mov dword ptr [r13], eax
	mov rax, qword ptr [rsp]
.Ltmp16516:
	lock inc	qword ptr [rax + 1032]
	mov rcx, qword ptr [rsp + 16]
.Ltmp16517:
	mov qword ptr [rcx + 8], 0
	xor eax, eax
	mov byte ptr [rcx], al
.Ltmp16518:
	mov rax, qword ptr [rsp + 96]
.Ltmp16519:
	lock dec	qword ptr [rax]
.Ltmp16520:
	jne .LBB157_182
.Ltmp16521:
.LBB157_181:
	#MEMBARRIER
	mov rdi, qword ptr [rsp + 152]
.Ltmp16522:
	call alloc::sync::Arc<T,A>::drop_slow
.Ltmp16523:
.LBB157_182:
	mov rdx, qword ptr [rsp + 168]
.Ltmp16524:
	mov rax, qword ptr [rdx + 8]
.Ltmp16525:
	lea rcx, [rax - 1]
.Ltmp16526:
	mov qword ptr [rdx + 8], rcx
.Ltmp16527:
	cmp rax, 1
	jne .LBB157_184
	mov rsi, -1
.Ltmp16529:
	xchg qword ptr [rdx], rsi
.Ltmp16530:
	cmp rsi, -1
	jne .LBB157_243
.Ltmp16531:
.LBB157_184:
	add rsp, 248
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
.Ltmp16532:
.LBB157_185:
	.cfi_def_cfa_offset 304
	mov byte ptr [r13 + r14 + 320], cl
.Ltmp16533:
	test r15, r15
	jne .LBB157_170
	jmp .LBB157_171
.Ltmp16534:
.LBB157_186:
	mov byte ptr [r13 + r14 + 320], cl
.Ltmp16535:
	test r15, r15
	jne .LBB157_178
	jmp .LBB157_179
.Ltmp16536:
.LBB157_187:
	mov r8, qword ptr [rsp + 96]
.Ltmp16537:
	lock inc	qword ptr [r8]
.Ltmp16538:
	jle .LBB157_256
.Ltmp16539:
	lea rcx, [rsp + 32]
	lea r9, [rsp + 176]
	mov rdi, qword ptr [rsp]
	mov rsi, r13
	mov rdx, r14
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::create_layer_concurrent_generic
	mov qword ptr [rsp + 112], rax
	mov rbx, qword ptr [rsp + 80]
	mov rsi, qword ptr [rsp + 88]
.Ltmp16540:
	xor r15d, r15d
.Ltmp16541:
	xchg qword ptr [r13 + 8*r14 + 344], r15
.Ltmp16542:
	test r15, r15
.Ltmp16543:
	je .LBB157_231
.Ltmp16544:
	mov rax, qword ptr [rsp]
.Ltmp16545:
	mov rax, qword ptr [rax + 8*rsi]
.Ltmp16546:
	test rax, rax
	je .LBB157_245
.Ltmp16548:
.LBB157_191:
	lea r12, [rax + rbx]
.Ltmp16549:
	movzx eax, byte ptr [rax + rbx + 128]
.Ltmp16550:
	test al, al
	je .LBB157_246
.Ltmp16551:
	mov qword ptr [rsp + 24], r12
.Ltmp16552:
	mov rbx, qword ptr [r12]
	test rbx, rbx
	jne .LBB157_224
.Ltmp16553:
.LBB157_193:
	mov rax, qword ptr [rsp]
	mov r12, qword ptr [rax + 952]
.Ltmp16554:
	mov rbx, r12
	shl rbx, 5
	mov rax, r12
	shr rax, 59
	sete al
	movabs rcx, 9223372036854775801
.Ltmp16555:
	cmp rbx, rcx
	setb cl
.Ltmp16556:
	test al, cl
	jne .LBB157_200
.Ltmp16557:
	call alloc::raw_vec::capacity_overflow
	jmp .LBB157_256
.Ltmp16558:
.LBB157_195:
	xor r14d, r14d
	xor r15d, r15d
.Ltmp16559:
	jmp .LBB157_165
.Ltmp16560:
.LBB157_196:
	xor r14d, r14d
	xor r15d, r15d
.Ltmp16561:
	jmp .LBB157_173
.Ltmp16562:
.LBB157_197:
	mov rcx, qword ptr [rsp + 16]
.Ltmp16563:
	mov byte ptr [rcx + 1], al
	mov al, 1
.Ltmp16564:
	mov byte ptr [rcx], al
.Ltmp16565:
	mov rax, qword ptr [rsp + 96]
.Ltmp16566:
	lock dec	qword ptr [rax]
.Ltmp16567:
	je .LBB157_181
	jmp .LBB157_182
.Ltmp16568:
.LBB157_198:
	xor r15d, r15d
.Ltmp16569:
	jmp .LBB157_165
.Ltmp16570:
.LBB157_199:
	xor r15d, r15d
.Ltmp16571:
	jmp .LBB157_173
.Ltmp16572:
.LBB157_200:
	test rbx, rbx
	je .LBB157_221
.Ltmp16573:
	mov esi, 8
	mov rdi, rbx
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
	mov qword ptr [rsp + 8], rax
.Ltmp16575:
	test rax, rax
	jne .LBB157_222
.Ltmp16576:
	mov edi, 8
	mov rsi, rbx
	call alloc::raw_vec::handle_error
.Ltmp15946:
	jmp .LBB157_256
.Ltmp16577:
.LBB157_203:
	lock inc	qword ptr [r15 - 16]
.Ltmp16578:
	jle .LBB157_256
.Ltmp16579:
	lea rbx, [r15 - 16]
.Ltmp16580:
	mov qword ptr [rsp + 216], rbx
.Ltmp16581:
	mov rax, qword ptr [rsp + 136]
.Ltmp16582:
	mov qword ptr [r13 + 8*r14 + 344], rax
.Ltmp16583:
	lea eax, [rbp + 512]
	mov ecx, ebp
	add ecx, 8
	test bpl, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	cmovne ecx, eax
	and ecx, esi
	mov dword ptr [r13], ecx
.Ltmp16584:
	mov rax, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 88]
.Ltmp16585:
	mov rax, qword ptr [rax + 8*rsi]
.Ltmp16586:
	test rax, rax
	mov rcx, qword ptr [rsp + 80]
	je .LBB157_248
.Ltmp16588:
	lea r14, [rax + rcx]
.Ltmp16589:
	movzx eax, byte ptr [rax + rcx + 128]
.Ltmp16590:
	test al, al
	je .LBB157_250
.Ltmp16591:
.LBB157_206:
	mov rax, qword ptr [rsp]
	mov rsi, qword ptr [rax + 952]
.Ltmp15915:
	mov rdi, r14
	call seize::raw::collector::LocalBatch::get_or_init
.Ltmp16593:
	cmp rax, -1
	je .LBB157_233
.Ltmp16594:
	mov r12, qword ptr [rax + 16]
.Ltmp16595:
	cmp r12, qword ptr [rax]
	jne .LBB157_210
	mov rdi, rax
	mov r13, rax
.Ltmp16598:
	call alloc::raw_vec::RawVec<T,A>::grow_one
	mov rax, r13
.Ltmp16599:
.LBB157_210:
	mov rcx, qword ptr [rax + 8]
.Ltmp16600:
	mov rdx, r12
	shl rdx, 5
.Ltmp16601:
	lea rsi, [rip + core::ops::function::FnOnce::call_once]
	mov qword ptr [rcx + rdx], rsi
	mov qword ptr [rcx + rdx + 8], r15
	mov qword ptr [rcx + rdx + 16], 0
	mov qword ptr [rcx + rdx + 24], rax
.Ltmp16602:
	inc r12
.Ltmp16603:
	mov qword ptr [rax + 16], r12
	mov rax, qword ptr [rsp]
.Ltmp16605:
	cmp r12, qword ptr [rax + 952]
	jb .LBB157_235
.Ltmp16606:
	mov rdi, qword ptr [rsp]
	mov rsi, r14
	call seize::raw::collector::Collector::try_retire
.Ltmp15920:
	jmp .LBB157_235
.Ltmp16607:
.LBB157_212:
	lock inc	qword ptr [r15 - 16]
	jle .LBB157_256
.Ltmp16609:
	lea rbx, [r15 - 16]
.Ltmp16610:
	mov qword ptr [rsp + 216], rbx
.Ltmp16611:
	mov rax, qword ptr [rsp + 136]
.Ltmp16612:
	mov qword ptr [r13 + 8*r14 + 344], rax
.Ltmp16613:
	lea eax, [rbp + 512]
	mov ecx, ebp
	add ecx, 8
	test bpl, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	cmovne ecx, eax
	and ecx, esi
	mov dword ptr [r13], ecx
.Ltmp16614:
	mov rax, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 88]
.Ltmp16615:
	mov rax, qword ptr [rax + 8*rsi]
.Ltmp16616:
	test rax, rax
	mov rcx, qword ptr [rsp + 80]
	je .LBB157_251
.Ltmp16618:
	lea r14, [rax + rcx]
.Ltmp16619:
	movzx eax, byte ptr [rax + rcx + 128]
.Ltmp16620:
	test al, al
	je .LBB157_253
.Ltmp16621:
.LBB157_215:
	mov rax, qword ptr [rsp]
	mov rsi, qword ptr [rax + 952]
.Ltmp15932:
	mov rdi, r14
	call seize::raw::collector::LocalBatch::get_or_init
.Ltmp16623:
	cmp rax, -1
	je .LBB157_233
.Ltmp16624:
	mov r12, qword ptr [rax + 16]
.Ltmp16625:
	cmp r12, qword ptr [rax]
	jne .LBB157_219
	mov rdi, rax
	mov r13, rax
.Ltmp16628:
	call alloc::raw_vec::RawVec<T,A>::grow_one
	mov rax, r13
.Ltmp16629:
.LBB157_219:
	mov rcx, qword ptr [rax + 8]
.Ltmp16630:
	mov rdx, r12
	shl rdx, 5
.Ltmp16631:
	lea rsi, [rip + core::ops::function::FnOnce::call_once]
	mov qword ptr [rcx + rdx], rsi
	mov qword ptr [rcx + rdx + 8], r15
	mov qword ptr [rcx + rdx + 16], 0
	mov qword ptr [rcx + rdx + 24], rax
.Ltmp16632:
	inc r12
.Ltmp16633:
	mov qword ptr [rax + 16], r12
	mov rax, qword ptr [rsp]
.Ltmp16635:
	cmp r12, qword ptr [rax + 952]
	jb .LBB157_235
.Ltmp16636:
	mov rdi, qword ptr [rsp]
	mov rsi, r14
	call seize::raw::collector::Collector::try_retire
.Ltmp15937:
	jmp .LBB157_235
.Ltmp16637:
.LBB157_221:
	mov eax, 8
	mov qword ptr [rsp + 8], rax
	xor r12d, r12d
.Ltmp16638:
.LBB157_222:
	mov edi, 32
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp16639:
	test rax, rax
	je .LBB157_247
.Ltmp16640:
	mov rbx, rax
	mov qword ptr [rax], r12
	mov rax, qword ptr [rsp + 8]
	mov qword ptr [rbx + 8], rax
	xorps xmm0, xmm0
	movups xmmword ptr [rbx + 16], xmm0
	mov rax, qword ptr [rsp + 24]
.Ltmp16641:
	mov qword ptr [rax], rbx
.Ltmp16642:
.LBB157_224:
	cmp rbx, -1
	je .LBB157_229
.Ltmp16643:
	mov r12, qword ptr [rbx + 16]
.Ltmp16644:
	cmp r12, qword ptr [rbx]
	jne .LBB157_227
.Ltmp15947:
	mov rdi, rbx
	call alloc::raw_vec::RawVec<T,A>::grow_one
.Ltmp16647:
.LBB157_227:
	mov rax, qword ptr [rbx + 8]
.Ltmp16648:
	mov rcx, r12
	shl rcx, 5
.Ltmp16649:
	lea rdx, [rip + core::ops::function::FnOnce::call_once]
	mov qword ptr [rax + rcx], rdx
	mov qword ptr [rax + rcx + 8], r15
	mov qword ptr [rax + rcx + 16], 0
	mov qword ptr [rax + rcx + 24], rbx
.Ltmp16650:
	inc r12
.Ltmp16651:
	mov qword ptr [rbx + 16], r12
	mov rax, qword ptr [rsp]
.Ltmp16653:
	cmp r12, qword ptr [rax + 952]
	jb .LBB157_231
.Ltmp16654:
	mov rdi, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 24]
	call seize::raw::collector::Collector::try_retire
	jmp .LBB157_231
.Ltmp16655:
.LBB157_229:
	lock dec	qword ptr [r15 - 16]
.Ltmp16656:
	jne .LBB157_231
.Ltmp16657:
	add r15, -16
.Ltmp16658:
	#MEMBARRIER
	mov rdi, r15
	call alloc::sync::Arc<T,A>::drop_slow
.Ltmp16659:
.LBB157_231:
	lea rdx, [rsp + 176]
.Ltmp16660:
	mov rdi, r13
	mov rsi, r14
	call masstree::leaf24::LeafNode24<S>::clear_ksuf
.Ltmp16661:
.Ltmp15957:
	mov byte ptr [r13 + r14 + 320], -128
	mov rax, qword ptr [rsp + 112]
.Ltmp16663:
	mov qword ptr [r13 + 8*r14 + 344], rax
.Ltmp16664:
	jmp .LBB157_180
.Ltmp16665:
.LBB157_233:
	lock dec	qword ptr [rbx]
	jne .LBB157_235
	#MEMBARRIER
	mov rdi, rbx
	call alloc::sync::Arc<T,A>::drop_slow
.Ltmp16667:
.LBB157_235:
	mov rax, qword ptr [rsp + 16]
	mov qword ptr [rax + 8], rbx
	mov byte ptr [rax], 0
	jmp .LBB157_182
.Ltmp16668:
.LBB157_236:
	mov rax, qword ptr fs:[0]
	lea rsi, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
	lea rdi, [rsp + 32]
	mov r14, rdx
.Ltmp16669:
	call seize::raw::tls::thread_id::Thread::init_slow
	mov rcx, qword ptr [rsp + 104]
	mov rdx, r14
	jmp .LBB157_2
.Ltmp16670:
.LBB157_237:
	mov rax, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 88]
.Ltmp16671:
	lea rdi, [rax + 8*rsi]
	add rdi, 472
	mov r14, rdx
	movaps xmmword ptr [rsp + 112], xmm0
	call seize::raw::tls::ThreadLocal<T>::initialize
	movaps xmm0, xmmword ptr [rsp + 112]
	mov rcx, qword ptr [rsp + 104]
	mov rdx, r14
	mov r15, rax
	jmp .LBB157_3
.Ltmp16673:
.LBB157_238:
	mov rdi, r15
	mov r14, rdx
	movaps xmmword ptr [rsp + 112], xmm0
	call seize::raw::tls::ThreadLocal<T>::write
	movaps xmm0, xmmword ptr [rsp + 112]
	mov rcx, qword ptr [rsp + 104]
	mov rdx, r14
	jmp .LBB157_4
.Ltmp16674:
.LBB157_239:
	mov qword ptr [rsp + 152], rcx
	lea rax, [rsp + 152]
	mov qword ptr [rsp + 216], rax
	lea rax, [rip + core::fmt::num::imp::<impl core::fmt::Display for u64>::fmt]
	mov qword ptr [rsp + 224], rax
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.156]
	mov qword ptr [rsp + 232], rcx
	mov qword ptr [rsp + 240], rax
.Ltmp16678:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.159]
.Ltmp16679:
	mov qword ptr [rsp + 32], rax
	mov qword ptr [rsp + 40], 2
	mov qword ptr [rsp + 64], 0
	lea rax, [rsp + 216]
.Ltmp16680:
	mov qword ptr [rsp + 48], rax
	mov qword ptr [rsp + 56], 2
.Ltmp16681:
.Ltmp15898:
	lea rsi, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.161]
	lea rdi, [rsp + 32]
	call core::panicking::panic_fmt
	jmp .LBB157_256
.Ltmp16683:
.LBB157_240:
	mov edi, 8
	mov esi, 24
	call alloc::alloc::handle_alloc_error
	jmp .LBB157_256
.Ltmp16684:
.LBB157_241:
	mov r14, rdx
.Ltmp16685:
	mov rdi, rdx
	mov rsi, rcx
	call masstree::key::Key::read_ikey_slow
.Ltmp16686:
.Ltmp15901:
	mov rdx, r14
	mov rcx, qword ptr [rsp + 104]
	jmp .LBB157_14
.Ltmp16687:
.LBB157_243:
	#MEMBARRIER
	mov rdi, qword ptr [rsp]
.Ltmp16688:
	add rsp, 248
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
	jmp seize::raw::collector::Collector::traverse
.Ltmp16689:
.LBB157_244:
	.cfi_def_cfa_offset 304
.Ltmp15939:
	lea rcx, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.197]
.Ltmp16690:
	mov rdi, rsi
.Ltmp16691:
	mov rsi, rax
.Ltmp16692:
	mov rdx, r8
.Ltmp16693:
	call core::slice::index::slice_index_fail
.Ltmp16694:
	jmp .LBB157_256
.Ltmp16695:
.LBB157_245:
	mov rax, qword ptr [rsp]
.Ltmp16696:
	lea rdi, [rax + 8*rsi]
.Ltmp16697:
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp16698:
.Ltmp15944:
	jmp .LBB157_191
.Ltmp16699:
.LBB157_246:
	mov rdi, r12
	call seize::raw::tls::ThreadLocal<T>::write
	mov qword ptr [rsp + 24], r12
.Ltmp16701:
	mov rbx, qword ptr [r12]
	test rbx, rbx
	jne .LBB157_224
	jmp .LBB157_193
.Ltmp16702:
.LBB157_247:
.Ltmp15951:
	mov edi, 8
	mov esi, 32
	call alloc::alloc::handle_alloc_error
.Ltmp15952:
	jmp .LBB157_256
.Ltmp16703:
.LBB157_248:
.Ltmp15913:
	mov rax, qword ptr [rsp]
.Ltmp16704:
	lea rdi, [rax + 8*rsi]
.Ltmp16705:
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp16706:
.Ltmp15914:
	mov rcx, qword ptr [rsp + 80]
.Ltmp16707:
	lea r14, [rax + rcx]
.Ltmp16708:
	movzx eax, byte ptr [rax + rcx + 128]
.Ltmp16709:
	test al, al
	jne .LBB157_206
.Ltmp16710:
.LBB157_250:
	mov rdi, r14
	call seize::raw::tls::ThreadLocal<T>::write
	jmp .LBB157_206
.Ltmp16711:
.LBB157_251:
.Ltmp15930:
	mov rax, qword ptr [rsp]
.Ltmp16712:
	lea rdi, [rax + 8*rsi]
.Ltmp16713:
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp16714:
.Ltmp15931:
	mov rcx, qword ptr [rsp + 80]
.Ltmp16715:
	lea r14, [rax + rcx]
.Ltmp16716:
	movzx eax, byte ptr [rax + rcx + 128]
.Ltmp16717:
	test al, al
	jne .LBB157_215
.Ltmp16718:
.LBB157_253:
	mov rdi, r14
	call seize::raw::tls::ThreadLocal<T>::write
	jmp .LBB157_215
.Ltmp16719:
.LBB157_254:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.188]
	mov qword ptr [rsp + 160], rax
.LBB157_255:
.Ltmp15966:
	mov esi, 24
	mov rdi, r14
	mov rdx, qword ptr [rsp + 160]
	call core::panicking::panic_bounds_check
.Ltmp16721:
.Ltmp15967:
.LBB157_256:
	ud2
.Ltmp16722:
.Ltmp15960:
	jmp .LBB157_269
.Ltmp16723:
.Ltmp15938:
	jmp .LBB157_260
.Ltmp15921:
.LBB157_260:
	mov r14, rax
.Ltmp16725:
	lock dec	qword ptr [rbx]
.Ltmp16726:
	jne .LBB157_276
.Ltmp16727:
	lea rax, [rsp + 216]
	jmp .LBB157_275
.Ltmp16728:
.Ltmp15965:
	jmp .LBB157_269
.Ltmp16729:
.Ltmp15953:
	mov r14, rax
.Ltmp16730:
	test r12, r12
	je .LBB157_272
.Ltmp16731:
	mov rdi, qword ptr [rsp + 8]
.Ltmp16732:
	call qword ptr [rip + mi_free@GOTPCREL]
.Ltmp16733:
	jmp .LBB157_272
.Ltmp16734:
.Ltmp15912:
	jmp .LBB157_271
.Ltmp16735:
.Ltmp15907:
	jmp .LBB157_269
.Ltmp16736:
.Ltmp15975:
	mov r14, rax
	jmp .LBB157_276
.Ltmp16737:
.Ltmp15904:
.LBB157_269:
	mov r14, rax
	jmp .LBB157_273
.Ltmp16738:
.Ltmp15972:
.LBB157_271:
	mov r14, rax
.Ltmp16739:
.LBB157_272:
	lea eax, [rbp + 512]
	mov ecx, ebp
	add ecx, 8
	test bpl, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	cmovne ecx, eax
	and ecx, esi
	mov dword ptr [r13], ecx
.Ltmp16740:
.LBB157_273:
	mov rax, qword ptr [rsp + 96]
.Ltmp16741:
	lock dec	qword ptr [rax]
.Ltmp16742:
	jne .LBB157_276
.Ltmp16743:
	lea rax, [rsp + 152]
.Ltmp16744:
.LBB157_275:
	#MEMBARRIER
	mov rdi, qword ptr [rax]
	call alloc::sync::Arc<T,A>::drop_slow
.Ltmp16745:
.LBB157_276:
.Ltmp15976:
	mov rdi, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 168]
	call core::ptr::drop_in_place<seize::guard::LocalGuard>
.Ltmp15977:
	mov rdi, r14
	call _Unwind_Resume@PLT
.Ltmp15978:
	call core::panicking::panic_in_cleanup
.Lfunc_end157:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert, .Lfunc_end157-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert","a",@progbits
	.p2align	2, 0x0
GCC_except_table157:
masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert:
.Lfunc_begin230:
	.cfi_startproc
	.cfi_personality 155, DW.ref.rust_eh_personality
	.cfi_lsda 27, .Lexception101
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
	sub rsp, 168
	.cfi_def_cfa_offset 224
	.cfi_offset rbx, -56
	.cfi_offset r12, -48
	.cfi_offset r13, -40
	.cfi_offset r14, -32
	.cfi_offset r15, -24
	.cfi_offset rbp, -16
	mov r15, rdx
	mov qword ptr [rsp], rsi
.Ltmp27813:
	test byte ptr fs:[seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF], 1
	mov qword ptr [rsp + 56], rdi
	mov qword ptr [rsp + 96], rcx
.Ltmp27814:
	je .LBB230_124
.Ltmp27815:
	mov rax, qword ptr fs:[0]
	lea rax, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
.Ltmp27816:
	mov rcx, qword ptr [rax + 24]
	mov qword ptr [rsp + 128], rcx
	movups xmm0, xmmword ptr [rax + 8]
	movaps xmmword ptr [rsp + 112], xmm0
.Ltmp27817:
.LBB230_2:
	mov rbx, qword ptr [rsp + 120]
	movaps xmm0, xmmword ptr [rsp + 112]
	mov rcx, qword ptr [rsp + 128]
	mov rax, qword ptr [rsp]
	mov qword ptr [rsp + 64], rcx
.Ltmp27820:
	mov rbp, qword ptr [rax + 8*rcx + 472]
.Ltmp27821:
	test rbp, rbp
	je .LBB230_125
.Ltmp27822:
.LBB230_3:
	shl rbx, 8
.Ltmp27823:
	movzx eax, byte ptr [rbp + rbx + 128]
.Ltmp27824:
	add rbp, rbx
.Ltmp27825:
	test al, al
	je .LBB230_126
.Ltmp27826:
.LBB230_4:
	mov rax, qword ptr [rbp + 8]
.Ltmp27827:
	lea rcx, [rax + 1]
.Ltmp27828:
	mov qword ptr [rbp + 8], rcx
.Ltmp27829:
	test rax, rax
	jne .LBB230_9
.Ltmp27830:
	movzx eax, byte ptr [rip + seize::raw::membarrier::linux::STRATEGY.0]
.Ltmp27831:
	cmp al, 2
	jne .LBB230_7
.Ltmp27832:
	xor eax, eax
.Ltmp27833:
	xchg qword ptr [rbp], rax
.Ltmp27834:
	jmp .LBB230_8
.Ltmp27835:
.LBB230_7:
	mov qword ptr [rbp], 0
.Ltmp27836:
.LBB230_8:
	#MEMBARRIER
.LBB230_9:
	mov rax, qword ptr [rsp]
	mov qword ptr [rsp + 112], rax
	movups xmmword ptr [rsp + 120], xmm0
	mov rax, qword ptr [rsp + 64]
	mov qword ptr [rsp + 136], rax
	mov qword ptr [rsp + 144], rbp
.Ltmp27837:
	bswap r15
.Ltmp27838:
	mov qword ptr [rsp + 104], 0
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.186]
	mov qword ptr [rsp + 88], rax
	mov qword ptr [rsp + 16], rbp
.Ltmp27839:
	mov qword ptr [rsp + 32], r15
	mov qword ptr [rsp + 160], rbx
.Ltmp27840:
	.p2align	4
.LBB230_10:
	mov rax, qword ptr [rsp]
.Ltmp27841:
	mov rax, qword ptr [rax + 1024]
.Ltmp27842:
.LBB230_11:
	mov rdi, rax
	lea rcx, [rax + 264]
	lea rdx, [rax + 560]
	mov qword ptr [rsp + 24], rax
	mov qword ptr [rsp + 8], rcx
	mov qword ptr [rsp + 72], rdx
	jmp .LBB230_14
.Ltmp27843:
.LBB230_12:
	mov rdi, r12
	call masstree::leaf24::LeafNode24<S>::wait_for_split
.Ltmp27844:
	.p2align	4
.LBB230_13:
	lea eax, [r15 + 512]
	mov ecx, r15d
	add ecx, 8
	test r15b, 4
	mov edx, -1342177792
	mov esi, -268435464
	cmove edx, esi
	cmovne ecx, eax
	and ecx, edx
	mov dword ptr [r12], ecx
	mov r15, qword ptr [rsp + 32]
.Ltmp27846:
	mov rdi, qword ptr [rsp + 24]
	mov rcx, qword ptr [rsp + 8]
	mov rdx, qword ptr [rsp + 72]
.Ltmp27847:
.LBB230_14:
	mov eax, dword ptr [rdi]
.Ltmp27848:
	test eax, eax
.Ltmp27849:
	mov rax, rcx
	cmovs rax, rdx
	mov rax, qword ptr [rax]
.Ltmp27850:
	test rax, rax
.Ltmp27851:
	jne .LBB230_11
.Ltmp27852:
.Ltmp27777:
	mov rsi, r15
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::reach_leaf_concurrent_generic
	mov r12, rax
.Ltmp27854:
	mov eax, dword ptr [rax]
.Ltmp27855:
	test al, 4
.Ltmp27856:
	jne .LBB230_18
	xor eax, eax
.Ltmp27858:
	jmp .LBB230_25
.Ltmp27859:
	.p2align	4
.LBB230_18:
	mov eax, dword ptr [r12]
	test al, 6
	je .LBB230_24
	xor eax, eax
	jmp .LBB230_21
	.p2align	4
.LBB230_20:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ecx, dword ptr [r12]
	test cl, 6
	je .LBB230_24
.LBB230_21:
	xor ecx, ecx
	.p2align	4
.LBB230_22:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB230_20
	cmp ecx, eax
	jbe .LBB230_22
	jmp .LBB230_20
	.p2align	4
.LBB230_24:
	#MEMBARRIER
	xor eax, eax
.Ltmp27868:
	.p2align	4
.LBB230_25:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27869:
	test dl, 1
.Ltmp27870:
	jne .LBB230_29
.Ltmp27871:
.LBB230_26:
	test rdx, rdx
.Ltmp27872:
	je .LBB230_55
.Ltmp27873:
	mov rcx, qword ptr [rdx + 128]
.Ltmp27874:
	cmp r15, rcx
.Ltmp27875:
	jb .LBB230_55
	inc rax
.Ltmp27877:
	mov r12, rdx
.Ltmp27878:
	cmp rax, 128
.Ltmp27879:
	jne .LBB230_25
	jmp .LBB230_10
	.p2align	4
.LBB230_29:
	xor ecx, ecx
	jmp .LBB230_31
.Ltmp27882:
	.p2align	4
.LBB230_30:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27883:
	mov ecx, 0
.Ltmp27884:
	test dl, 1
	je .LBB230_26
.Ltmp27885:
.LBB230_31:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27886:
	test dl, 1
.Ltmp27887:
	je .LBB230_30
.Ltmp27888:
	pause
.Ltmp27889:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27890:
	test dl, 1
.Ltmp27891:
	je .LBB230_30
.Ltmp27892:
	pause
.Ltmp27893:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27894:
	test dl, 1
.Ltmp27895:
	je .LBB230_30
.Ltmp27896:
	pause
.Ltmp27897:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27898:
	test dl, 1
.Ltmp27899:
	je .LBB230_30
.Ltmp27900:
	pause
.Ltmp27901:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27902:
	test dl, 1
.Ltmp27903:
	je .LBB230_30
.Ltmp27904:
	pause
.Ltmp27905:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27906:
	test dl, 1
.Ltmp27907:
	je .LBB230_30
.Ltmp27908:
	pause
.Ltmp27909:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27910:
	test dl, 1
.Ltmp27911:
	je .LBB230_30
.Ltmp27912:
	pause
.Ltmp27913:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27914:
	test dl, 1
.Ltmp27915:
	je .LBB230_30
.Ltmp27916:
	pause
.Ltmp27917:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27918:
	test dl, 1
.Ltmp27919:
	je .LBB230_30
.Ltmp27920:
	pause
.Ltmp27921:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27922:
	test dl, 1
.Ltmp27923:
	je .LBB230_30
.Ltmp27924:
	pause
.Ltmp27925:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27926:
	test dl, 1
.Ltmp27927:
	je .LBB230_30
.Ltmp27928:
	pause
.Ltmp27929:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27930:
	test dl, 1
.Ltmp27931:
	je .LBB230_30
.Ltmp27932:
	pause
.Ltmp27933:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27934:
	test dl, 1
.Ltmp27935:
	je .LBB230_30
.Ltmp27936:
	pause
.Ltmp27937:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27938:
	test dl, 1
.Ltmp27939:
	je .LBB230_30
.Ltmp27940:
	pause
.Ltmp27941:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27942:
	test dl, 1
.Ltmp27943:
	je .LBB230_30
.Ltmp27944:
	pause
.Ltmp27945:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27946:
	test dl, 1
.Ltmp27947:
	je .LBB230_30
.Ltmp27948:
	pause
.Ltmp27949:
	mov rdx, qword ptr [r12 + 544]
.Ltmp27950:
	test dl, 1
.Ltmp27951:
	je .LBB230_30
.Ltmp27952:
	mov edx, dword ptr [r12]
	test dl, 6
	je .LBB230_54
.Ltmp27953:
	xor edx, edx
	jmp .LBB230_51
	.p2align	4
.LBB230_50:
	and edx, 7
	lea edx, [2*rdx + 1]
	mov esi, dword ptr [r12]
	test sil, 6
	je .LBB230_54
.LBB230_51:
	xor esi, esi
	.p2align	4
.LBB230_52:
	mov edi, esi
	pause
	cmp esi, edx
	adc esi, 0
	cmp edi, edx
	jae .LBB230_50
	cmp esi, edx
	jbe .LBB230_52
	jmp .LBB230_50
.LBB230_54:
	#MEMBARRIER
	inc rcx
	cmp rcx, 1001
	jne .LBB230_31
	jmp .LBB230_30
.Ltmp27960:
	.p2align	4
.LBB230_55:
	mov ebp, dword ptr [r12]
	test bpl, 6
	je .LBB230_61
	xor eax, eax
	jmp .LBB230_58
	.p2align	4
.LBB230_57:
	and eax, 7
	lea eax, [2*rax + 1]
	mov ebp, dword ptr [r12]
	test bpl, 6
	je .LBB230_61
.LBB230_58:
	xor ecx, ecx
	.p2align	4
.LBB230_59:
	mov edx, ecx
	pause
	cmp ecx, eax
	adc ecx, 0
	cmp edx, eax
	jae .LBB230_57
	cmp ecx, eax
	jbe .LBB230_59
	jmp .LBB230_57
.Ltmp27967:
	.p2align	4
.LBB230_61:
	#MEMBARRIER
	lea rdi, [r12 + 64]
.Ltmp27969:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
	mov qword ptr [rsp + 80], rdi
.Ltmp27971:
	call rax
.Ltmp27972:
.Ltmp27780:
	mov r14, rax
	mov rbx, rdx
	xor ecx, ecx
	jmp .LBB230_64
	.p2align	4
.LBB230_63:
	and ecx, 7
	lea ecx, [2*rcx + 1]
.LBB230_64:
	mov r15d, dword ptr [r12]
	test r15b, 1
	jne .LBB230_66
	mov r13d, r15d
	or r13d, 3
	mov eax, r15d
	lock cmpxchg	dword ptr [r12], r13d
	je .LBB230_69
.LBB230_66:
	xor eax, eax
	.p2align	4
.LBB230_67:
	mov edx, eax
	pause
	cmp eax, ecx
	adc eax, 0
	cmp edx, ecx
	jae .LBB230_63
	cmp eax, ecx
	jbe .LBB230_67
	jmp .LBB230_63
.Ltmp27979:
	.p2align	4
.LBB230_69:
	#MEMBARRIER
	mov eax, dword ptr [r12]
.Ltmp27981:
	xor eax, ebp
	cmp eax, 3
	mov rbp, qword ptr [rsp + 16]
.Ltmp27982:
	ja .LBB230_13
.Ltmp27983:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
.Ltmp27781:
	mov rdi, qword ptr [rsp + 80]
.Ltmp27985:
	call rax
.Ltmp27986:
	xor rax, r14
	xor rdx, rbx
	or rdx, rax
.Ltmp27987:
	jne .LBB230_13
.Ltmp27988:
	mov rax, qword ptr [r12 + 544]
.Ltmp27989:
	test al, 1
	jne .LBB230_12
.Ltmp27990:
	test rax, rax
.Ltmp27991:
	je .LBB230_75
.Ltmp27992:
	mov rax, qword ptr [rax + 128]
.Ltmp27993:
	cmp qword ptr [rsp + 32], rax
	jae .LBB230_13
.Ltmp27994:
.LBB230_75:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_load::FUNC]
	mov rdi, qword ptr [rsp + 80]
.Ltmp27996:
	call rax
.Ltmp27997:
.Ltmp27784:
	mov rdi, rax
	and rdi, 31
.Ltmp27998:
	je .LBB230_87
.Ltmp27999:
	mov ecx, 5
	xor r14d, r14d
	jmp .LBB230_80
.Ltmp28000:
	.p2align	4
.LBB230_78:
	ja .LBB230_86
.Ltmp28001:
.LBB230_79:
	inc r14
.Ltmp28002:
	add rcx, 5
	cmp rdi, r14
.Ltmp28003:
	je .LBB230_88
.Ltmp28004:
.LBB230_80:
	mov rsi, rax
	shrd rsi, rdx, cl
	mov rbx, rdx
	shr rbx, cl
	test cl, 64
	cmove rbx, rsi
	and ebx, 31
.Ltmp28005:
	cmp rbx, 23
	ja .LBB230_133
.Ltmp28006:
	mov rsi, qword ptr [r12 + 8*rbx + 128]
.Ltmp28007:
	cmp rsi, qword ptr [rsp + 32]
	jne .LBB230_78
.Ltmp28008:
	movzx r8d, byte ptr [r12 + rbx + 320]
.Ltmp28009:
	mov rsi, qword ptr [r12 + 8*rbx + 344]
.Ltmp28010:
	test rsi, rsi
.Ltmp28011:
	je .LBB230_79
	test r8b, r8b
	js .LBB230_86
	cmp r8b, 8
	jne .LBB230_78
.Ltmp28014:
	mov r14, qword ptr [r12 + 8*rbx + 344]
.Ltmp28015:
	test r14, r14
.Ltmp28016:
	je .LBB230_13
	jmp .LBB230_99
.Ltmp28017:
.LBB230_86:
	cmp edi, 23
	ja .LBB230_89
	jmp .LBB230_90
.Ltmp28018:
.LBB230_87:
	xor r14d, r14d
.Ltmp28019:
	jmp .LBB230_90
.Ltmp28020:
.LBB230_88:
	mov r14, rdi
.Ltmp28021:
	cmp edi, 23
	jbe .LBB230_90
.LBB230_89:
.Ltmp27798:
	mov rdi, qword ptr [rsp]
.Ltmp28023:
	mov rsi, r12
	mov rdx, r12
.Ltmp28024:
	mov ecx, r13d
	mov r15, qword ptr [rsp + 32]
.Ltmp28025:
	mov r8, r15
	lea r9, [rsp + 112]
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
.Ltmp28026:
	jmp .LBB230_97
.Ltmp28027:
.LBB230_90:
	mov r9, rdx
	shr r9, 56
.Ltmp28028:
	mov rbx, r9
	and rbx, 31
.Ltmp28029:
	jne .LBB230_120
.Ltmp28030:
	mov rcx, qword ptr [r12 + 552]
.Ltmp28031:
	test rcx, rcx
	mov rbp, qword ptr [rsp + 16]
.Ltmp28032:
	je .LBB230_119
.Ltmp28033:
	mov rcx, qword ptr [r12 + 128]
.Ltmp28034:
	cmp rcx, qword ptr [rsp + 32]
.Ltmp28035:
	je .LBB230_119
.Ltmp28036:
	add rdi, -23
.Ltmp28037:
	mov ecx, 115
	xor r8d, r8d
.Ltmp28038:
	.p2align	4
.LBB230_94:
	cmp rdi, r8
.Ltmp28039:
	je .LBB230_96
.Ltmp28040:
	mov rsi, rax
	shrd rsi, rdx, cl
	mov rbx, rdx
	shr rbx, cl
	test cl, 64
	cmove rbx, rsi
.Ltmp28041:
	dec r8
.Ltmp28042:
	add rcx, -5
	and rbx, 31
.Ltmp28043:
	je .LBB230_94
	jmp .LBB230_108
.Ltmp28044:
.LBB230_96:
	mov rdi, qword ptr [rsp]
.Ltmp28045:
	mov rsi, r12
	mov rdx, r12
.Ltmp28046:
	mov ecx, r13d
	mov r15, qword ptr [rsp + 32]
.Ltmp28047:
	mov r8, r15
	lea r9, [rsp + 112]
	call masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::handle_leaf_split_generic
.Ltmp28048:
.Ltmp27801:
.LBB230_97:
	cmp al, 6
	mov rbp, qword ptr [rsp + 16]
	mov rdi, qword ptr [rsp + 24]
	mov rcx, qword ptr [rsp + 8]
	mov rdx, qword ptr [rsp + 72]
	je .LBB230_14
	mov rcx, qword ptr [rsp + 56]
.Ltmp28050:
	mov byte ptr [rcx + 8], al
	mov qword ptr [rcx], 2
.Ltmp28051:
	jmp .LBB230_116
.Ltmp28052:
.LBB230_108:
	mov qword ptr [rsp + 24], r9
.Ltmp28053:
	mov qword ptr [rsp + 8], rdx
.Ltmp28054:
	mov r13, rax
.Ltmp28055:
	neg r8
	mov qword ptr [rsp + 104], r8
.Ltmp28056:
.LBB230_109:
	mov edi, 8
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp28057:
	test rax, rax
	je .LBB230_127
.Ltmp28058:
	mov rcx, qword ptr [rsp + 96]
	mov qword ptr [rax], rcx
.Ltmp28059:
	cmp ebx, 24
	mov r10, qword ptr [rsp + 24]
	jae .LBB230_132
.Ltmp28060:
	mov r11, r13
	mov rcx, qword ptr [rsp + 32]
.Ltmp28062:
	mov qword ptr [r12 + 8*rbx + 128], rcx
.Ltmp28063:
	mov qword ptr [r12 + 8*rbx + 344], rax
.Ltmp28064:
	mov byte ptr [r12 + rbx + 320], 8
.Ltmp28065:
	mov rcx, qword ptr [rsp + 104]
.Ltmp28066:
	test rcx, rcx
	je .LBB230_113
.Ltmp28067:
	mov al, 23
.Ltmp28068:
	sub al, cl
	movzx eax, al
	lea ecx, [rax + 4*rax]
	add cl, 5
	mov rax, r11
	mov rdx, qword ptr [rsp + 8]
	shrd rax, rdx, cl
	mov rsi, rdx
	shr rsi, cl
	test cl, 64
	cmove rsi, rax
	xor r10d, esi
	and r10d, 31
.Ltmp28069:
	mov rax, r10
	shl rax, cl
	xor edi, edi
.Ltmp28070:
	test cl, 64
.Ltmp28071:
	mov rsi, rax
	cmovne rsi, rdi
	shld rdi, r10, cl
.Ltmp28072:
	test cl, 64
.Ltmp28073:
	cmovne rdi, rax
	shl r10, 56
.Ltmp28074:
	or r10, rdi
	xor rdx, r10
.Ltmp28075:
	xor r11, rsi
.Ltmp28076:
	mov r10, rdx
	shr r10, 56
	jmp .LBB230_114
.Ltmp28077:
.LBB230_113:
	mov rdx, qword ptr [rsp + 8]
.Ltmp28078:
.LBB230_114:
	lea ecx, [r14 + 4*r14]
	add cl, 5
.Ltmp28079:
	mov edi, 31
	shl rdi, cl
	mov r8d, 31
	xor esi, esi
.Ltmp28080:
	test cl, 64
.Ltmp28081:
	mov rax, rdi
	cmovne rax, rsi
	xor ebx, ebx
.Ltmp28082:
	shld rbx, r8, cl
.Ltmp28083:
	test cl, 64
.Ltmp28084:
	cmovne rbx, rdi
	and r10d, 31
.Ltmp28085:
	mov r9, r10
	shl r9, cl
.Ltmp28086:
	test cl, 64
.Ltmp28087:
	mov rdi, r9
	cmovne rdi, rsi
	xor r8d, r8d
	shld r8, r10, cl
.Ltmp28088:
	test cl, 64
.Ltmp28089:
	cmovne r8, r9
.Ltmp28090:
	mov r10, -1
	shl r10, cl
	test cl, 64
	cmove rsi, r10
	mov rcx, -1
	cmove r10, rcx
.Ltmp28091:
	mov rcx, r11
	add rcx, 1
	mov r9, rdx
	adc r9, 0
	shld rdx, r11, 5
.Ltmp28092:
	and rdx, r10
.Ltmp28093:
	not r10
.Ltmp28094:
	shl r11, 5
.Ltmp28095:
	and r11, rsi
.Ltmp28096:
	not rsi
.Ltmp28097:
	and rcx, rsi
	or rcx, rdi
	and r9, r10
	or r9, r8
	not rbx
	not rax
	and rdx, rbx
	or rdx, r9
	and r11, rax
	or r11, rcx
.Ltmp28098:
	mov rax, qword ptr [rip + portable_atomic::imp::atomic128::x86_64::atomic_store::FUNC]
.Ltmp27805:
	mov rdi, qword ptr [rsp + 80]
.Ltmp28100:
	mov rsi, r11
.Ltmp28101:
	call rax
.Ltmp28102:
.Ltmp27806:
	lea eax, [r15 + 512]
	mov ecx, r15d
	add ecx, 8
	test r15b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	cmovne ecx, eax
	and ecx, esi
	mov dword ptr [r12], ecx
	mov rax, qword ptr [rsp]
.Ltmp28104:
	lock inc	qword ptr [rax + 1032]
	mov rax, qword ptr [rsp + 56]
.Ltmp28105:
	mov qword ptr [rax], 0
.Ltmp28106:
.LBB230_116:
	mov rax, qword ptr [rbp + 8]
.Ltmp28107:
	lea rcx, [rax - 1]
.Ltmp28108:
	mov qword ptr [rbp + 8], rcx
.Ltmp28109:
	cmp rax, 1
	jne .LBB230_118
	mov rsi, -1
.Ltmp28111:
	xchg qword ptr [rbp], rsi
.Ltmp28112:
	cmp rsi, -1
	jne .LBB230_128
.Ltmp28113:
.LBB230_118:
	add rsp, 168
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
.Ltmp28114:
.LBB230_119:
	.cfi_def_cfa_offset 224
	mov qword ptr [rsp + 24], r9
	mov qword ptr [rsp + 8], rdx
.Ltmp28116:
	mov r13, rax
.Ltmp28117:
	xor ebx, ebx
.Ltmp28118:
	jmp .LBB230_109
.Ltmp28119:
.LBB230_120:
	mov qword ptr [rsp + 24], r9
.Ltmp28120:
	mov qword ptr [rsp + 8], rdx
.Ltmp28121:
	mov r13, rax
.Ltmp28122:
	mov rbp, qword ptr [rsp + 16]
	jmp .LBB230_109
.Ltmp28123:
.LBB230_99:
	mov r13, qword ptr [r14]
.Ltmp28124:
	mov edi, 8
	mov esi, 8
	call qword ptr [rip + mi_malloc_aligned@GOTPCREL]
.Ltmp28125:
	test rax, rax
	je .LBB230_129
.Ltmp28126:
	mov rcx, qword ptr [rsp + 96]
	mov qword ptr [rax], rcx
.Ltmp28127:
	mov qword ptr [r12 + 8*rbx + 344], rax
.Ltmp28128:
	lea eax, [r15 + 512]
.Ltmp28129:
	mov ecx, r15d
	add ecx, 8
	test r15b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	cmovne ecx, eax
	and ecx, esi
	mov dword ptr [r12], ecx
.Ltmp28130:
	mov rax, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 64]
.Ltmp28131:
	mov rax, qword ptr [rax + 8*rsi]
.Ltmp28132:
	test rax, rax
	je .LBB230_130
.Ltmp28134:
.LBB230_101:
	mov rcx, qword ptr [rsp + 160]
.Ltmp28135:
	lea rbx, [rax + rcx]
.Ltmp28136:
	movzx eax, byte ptr [rax + rcx + 128]
.Ltmp28137:
	test al, al
	je .LBB230_131
.Ltmp28138:
.LBB230_102:
	mov rax, qword ptr [rsp]
.Ltmp28139:
	mov rsi, qword ptr [rax + 952]
.Ltmp27788:
	mov rdi, rbx
	call seize::raw::collector::LocalBatch::get_or_init
.Ltmp28140:
	cmp rax, -1
	je .LBB230_122
.Ltmp28141:
	mov r15, qword ptr [rax + 16]
.Ltmp28142:
	cmp r15, qword ptr [rax]
	jne .LBB230_106
	mov rdi, rax
	mov r12, rax
.Ltmp28145:
	call alloc::raw_vec::RawVec<T,A>::grow_one
	mov rax, r12
.Ltmp28146:
.LBB230_106:
	mov rcx, qword ptr [rax + 8]
.Ltmp28147:
	mov rdx, r15
	shl rdx, 5
.Ltmp28148:
	lea rsi, [rip + core::ops::function::FnOnce::call_once]
	mov qword ptr [rcx + rdx], rsi
	mov qword ptr [rcx + rdx + 8], r14
	mov qword ptr [rcx + rdx + 16], 0
	mov qword ptr [rcx + rdx + 24], rax
.Ltmp28149:
	inc r15
.Ltmp28150:
	mov qword ptr [rax + 16], r15
	mov rax, qword ptr [rsp]
.Ltmp28152:
	cmp r15, qword ptr [rax + 952]
	jb .LBB230_123
.Ltmp28153:
	mov rdi, qword ptr [rsp]
	mov rsi, rbx
	call seize::raw::collector::Collector::try_retire
.Ltmp27793:
	jmp .LBB230_123
.Ltmp28154:
.LBB230_122:
	mov rdi, r14
	call qword ptr [rip + mi_free@GOTPCREL]
.Ltmp28155:
.LBB230_123:
	mov rax, qword ptr [rsp + 56]
	mov qword ptr [rax], 1
	mov qword ptr [rax + 8], r13
	jmp .LBB230_116
.Ltmp28156:
.LBB230_124:
	mov rax, qword ptr fs:[0]
	lea rsi, [rax + seize::raw::tls::thread_id::THREAD::{{constant}}::{{closure}}::__RUST_STD_INTERNAL_VAL@TPOFF]
	lea rdi, [rsp + 112]
	call seize::raw::tls::thread_id::Thread::init_slow
	jmp .LBB230_2
.Ltmp28157:
.LBB230_125:
	mov rax, qword ptr [rsp]
	mov rsi, qword ptr [rsp + 64]
.Ltmp28158:
	lea rdi, [rax + 8*rsi]
	add rdi, 472
	movaps xmmword ptr [rsp + 32], xmm0
	call seize::raw::tls::ThreadLocal<T>::initialize
	movaps xmm0, xmmword ptr [rsp + 32]
	mov rbp, rax
	jmp .LBB230_3
.Ltmp28160:
.LBB230_126:
	mov rdi, rbp
	movaps xmmword ptr [rsp + 32], xmm0
	call seize::raw::tls::ThreadLocal<T>::write
	movaps xmm0, xmmword ptr [rsp + 32]
	jmp .LBB230_4
.Ltmp28161:
.LBB230_127:
.Ltmp27807:
	mov edi, 8
	mov esi, 8
	call alloc::alloc::handle_alloc_error
.Ltmp27808:
	jmp .LBB230_134
.Ltmp28162:
.LBB230_128:
	#MEMBARRIER
	mov rdi, qword ptr [rsp]
.Ltmp28163:
	add rsp, 168
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
	jmp seize::raw::collector::Collector::traverse
.Ltmp28164:
.LBB230_129:
	.cfi_def_cfa_offset 224
.Ltmp27795:
	mov edi, 8
	mov esi, 8
	call alloc::alloc::handle_alloc_error
.Ltmp27796:
	jmp .LBB230_134
.Ltmp28165:
.LBB230_130:
.Ltmp27786:
	mov rax, qword ptr [rsp]
.Ltmp28166:
	lea rdi, [rax + 8*rsi]
.Ltmp28167:
	call seize::raw::tls::ThreadLocal<T>::initialize
.Ltmp28168:
.Ltmp27787:
	jmp .LBB230_101
.Ltmp28169:
.LBB230_131:
	mov rdi, rbx
	call seize::raw::tls::ThreadLocal<T>::write
	jmp .LBB230_102
.Ltmp28170:
.LBB230_132:
	lea rax, [rip + .Lanon.79d957a09dfce7ea1bb60de69dcd69e0.188]
	mov qword ptr [rsp + 88], rax
.Ltmp28171:
.LBB230_133:
.Ltmp27803:
	mov esi, 24
	mov rdi, rbx
	mov rdx, qword ptr [rsp + 88]
	mov rbp, qword ptr [rsp + 16]
	call core::panicking::panic_bounds_check
.Ltmp28172:
.Ltmp27804:
.LBB230_134:
	ud2
.Ltmp28173:
.Ltmp27794:
	mov rbx, rax
	jmp .LBB230_142
.Ltmp28174:
.Ltmp27797:
	mov rbx, rax
	mov rbp, qword ptr [rsp + 16]
	jmp .LBB230_141
.Ltmp27785:
	jmp .LBB230_140
.Ltmp28176:
.Ltmp27802:
	mov rbx, rax
	mov rbp, qword ptr [rsp + 16]
	jmp .LBB230_142
.Ltmp28177:
.Ltmp27809:
.LBB230_140:
	mov rbx, rax
.Ltmp28178:
.LBB230_141:
	lea eax, [r15 + 512]
	mov ecx, r15d
	add ecx, 8
	test r15b, 4
	mov edx, -268435464
	mov esi, -1342177792
	cmove esi, edx
	cmovne ecx, eax
	and ecx, esi
	mov dword ptr [r12], ecx
.Ltmp28179:
.LBB230_142:
.Ltmp27810:
	mov rdi, qword ptr [rsp]
	mov rsi, rbp
	call core::ptr::drop_in_place<seize::guard::LocalGuard>
.Ltmp27811:
	mov rdi, rbx
	call _Unwind_Resume@PLT
.Ltmp27812:
	call core::panicking::panic_in_cleanup
.Lfunc_end230:
	.size	masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert, .Lfunc_end230-masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert
	.cfi_endproc
.section ".gcc_except_table.masstree::tree::generic::<impl masstree::tree::MassTreeGeneric<S,L,A>>::insert","a",@progbits
	.p2align	2, 0x0
