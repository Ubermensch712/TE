using Random, Statistics, LinearAlgebra

# ==========================================
# 1. GENERADOR DEL SISTEMA 2 (Rigor Papana)
# ==========================================
function generar_sistema_2(n; burn_in=1000)
    total_n = n + burn_in
    x1, x2, x3 = zeros(total_n), zeros(total_n), zeros(total_n)
    e = randn(3, total_n)
    for t in 3:total_n
        x1[t] = 0.7*x1[t-1] + e[1,t]
        x2[t] = 0.3*x2[t-1] + 0.5*x2[t-2]*x1[t-1] + e[2,t]
        x3[t] = 0.3*x3[t-1] + 0.5*x3[t-2]*x1[t-1] + e[3,t]
    end
    return x1[burn_in+1:end], x2[burn_in+1:end], x3[burn_in+1:end]
end

# ==========================================
# 2. IMPLEMENTACIÓN PSTE (PARCIAL SIMBÓLICA)
# ==========================================
@inline function simbolizar!(s, x)
    @inbounds for i in 2:length(x)
        s[i-1] = x[i] > x[i-1] ? 1 : 0
    end
end

function fast_pste(s_src, s_tar, s_cnd)
    n = length(s_tar) - 1
    c_f, c_c, c_xyz, c_yz = zeros(Int, 16), zeros(Int, 8), zeros(Int, 8), zeros(Int, 4)
    @inbounds for i in 1:n
        yf, yp, xp, zp = s_tar[i+1], s_tar[i], s_src[i], s_cnd[i]
        c_f[1 + yf + 2*yp + 4*zp + 8*xp] += 1
        c_c[1 + yf + 2*yp + 4*zp] += 1
        c_xyz[1 + xp + 2*yp + 4*zp] += 1
        c_yz[1 + yp + 2*zp] += 1
    end
    H(counts) = (tot = sum(counts); tot == 0 ? 0.0 : -sum([c > 0 ? (p=c/tot; p*log2(p)) : 0.0 for c in counts]))
    return H(c_c) + H(c_xyz) - H(c_f) - H(c_yz)
end

# ==========================================
# 3. IMPLEMENTACIÓN CGCI (GRANGER LINEAL)
# ==========================================
function fast_cgci(src, tar, cnd, p=2)
    N = length(tar)
    Y = tar[p+1:N]
    n_obs = length(Y)
    X_R = ones(n_obs, 1 + 2*p)
    for i in 1:p
        X_R[:, 1+i] = tar[p+1-i : N-i]
        X_R[:, 1+p+i] = cnd[p+1-i : N-i]
    end
    X_F = hcat(X_R, zeros(n_obs, p))
    for i in 1:p
        X_F[:, 1+2*p+i] = src[p+1-i : N-i]
    end
    var_R = var(Y - X_R * (X_R \ Y))
    var_F = var(Y - X_F * (X_F \ Y))
    return (var_F > 0 && var_R > var_F) ? log(var_R / var_F) : 0.0
end

# ==========================================
# 4. PRUEBAS DE SIGNIFICANCIA (SURROGATES)
# ==========================================
function test_pste(s_src, s_tar, s_cnd; n_surr=40)
    val = fast_pste(s_src, s_tar, s_cnd)
    s_shuf = copy(s_src); count = 0
    for _ in 1:n_surr
        shuffle!(s_shuf)
        fast_pste(s_shuf, s_tar, s_cnd) >= val && (count += 1)
    end
    return (count / n_surr) < 0.05
end

function test_cgci(src, tar, cnd; n_surr=40)
    val = fast_cgci(src, tar, cnd)
    src_shuf = copy(src); count = 0
    for _ in 1:n_surr
        shuffle!(src_shuf)
        fast_cgci(src_shuf, tar, cnd) >= val && (count += 1)
    end
    return (count / n_surr) < 0.05
end

# ==========================================
# 5. FUNCIÓN DE BENCHMARKING (ENCAPSULADA)
# ==========================================
function ejecutar_benchmarking()
    Random.seed!(123)
    n_realizaciones = 100
    tamanos_n = [512, 2048]

    # Contenedores para organizar los resultados
    # Guardamos [n512_pste, n2048_pste] y [n512_cgci, n2048_cgci]
    tabla_pste = zeros(Int, 2, 6)
    tabla_cgci = zeros(Int, 2, 6)

    println("\n" * "="^70)
    println(" EJECUTANDO SIMULACIÓN DE BENCHMARKING (Sistema 2)")
    println("="^70)

    for (idx, n) in enumerate(tamanos_n)
        print("Simulando n = $n ... ")
        for i in 1:n_realizaciones
            # Dentro de la función, estas variables son locales y seguras
            x_sim1, x_sim2, x_sim3 = generar_sistema_2(n)
            
            s1, s2, s3 = zeros(Int, n), zeros(Int, n), zeros(Int, n)
            simbolizar!(s1, x_sim1); simbolizar!(s2, x_sim2); simbolizar!(s3, x_sim3)
            
            # Acumulamos resultados PSTE
            test_pste(s1, s2, s3) && (tabla_pste[idx, 1] += 1)
            test_pste(s2, s1, s3) && (tabla_pste[idx, 2] += 1)
            test_pste(s2, s3, s1) && (res = test_pste(s2, s3, s1); tabla_pste[idx, 3] += res ? 1 : 0)
            test_pste(s3, s2, s1) && (tabla_pste[idx, 4] += 1)
            test_pste(s1, s3, s2) && (tabla_pste[idx, 5] += 1)
            test_pste(s3, s1, s2) && (tabla_pste[idx, 6] += 1)

            # Acumulamos resultados CGCI
            test_cgci(x_sim1, x_sim2, x_sim3) && (tabla_cgci[idx, 1] += 1)
            test_cgci(x_sim2, x_sim1, x_sim3) && (tabla_cgci[idx, 2] += 1)
            test_cgci(x_sim2, x_sim3, x_sim1) && (tabla_cgci[idx, 3] += 1)
            test_cgci(x_sim3, x_sim2, x_sim1) && (tabla_cgci[idx, 4] += 1)
            test_cgci(x_sim1, x_sim3, x_sim2) && (tabla_cgci[idx, 5] += 1)
            test_cgci(x_sim3, x_sim1, x_sim2) && (tabla_cgci[idx, 6] += 1)
        end
        println("Hecho.")
    end

    # IMPRESIÓN ORGANIZADA
    println("-"^70)
    println(" n     | X1->X2 | X2->X1 | X2->X3 | X3->X2 | X1->X3 | X3->X1")
    println("-"^70)
    println("PSTE (m=2)")
    for (i, n) in enumerate(tamanos_n)
        println("$(rpad(n, 6)) |   $(tabla_pste[i,1])   |   $(tabla_pste[i,2])   |   $(tabla_pste[i,3])   |   $(tabla_pste[i,4])   |   $(tabla_pste[i,5])   |   $(tabla_pste[i,6])")
    end
    println("-"^70)
    println("CGCI (P=2)")
    for (i, n) in enumerate(tamanos_n)
        println("$(rpad(n, 6)) |   $(tabla_cgci[i,1])   |   $(tabla_cgci[i,2])   |   $(tabla_cgci[i,3])   |   $(tabla_cgci[i,4])   |   $(tabla_cgci[i,5])   |   $(tabla_cgci[i,6])")
    end
    println("="^70)
end

# Arrancamos la simulación
ejecutar_benchmarking()