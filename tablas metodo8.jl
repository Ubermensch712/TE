println("cargando paquetes en la memoria, por favor espera...")
flush(stdout)

using Random, Statistics, LinearAlgebra, Printf

println("paquetes cargados con exito. compilando funciones...")
flush(stdout)

# ==========================================
# modulo 1: generadores de sistenas
# ==========================================

function generar_igarch(n; a0=0.2, a1=0.9, b1=0.1)
    z, s2, err = zeros(n), zeros(n), randn(n)
    s2[1] = a0
    for t in 2:n
        s2[t] = a0 + a1 * z[t-1]^2 + b1 * s2[t-1]
        z[t] = sqrt(s2[t]) * err[t]
    end
    return z
end

function generar_datos_sistema8(n; burn_in=1000, g=1.0)
    total_n = n + burn_in
    x1, x2, x3 = zeros(total_n), zeros(total_n), zeros(total_n)
    e = randn(3, total_n)
    
    for t in 3:total_n
        x1[t] = 0.7*x1[t-1] + e[1,t]
        x2[t] = 0.3*x2[t-1] + 0.5*x2[t-2]*x1[t-1] + e[2,t]
        x3[t] = 0.3*x3[t-1] + 0.5*x3[t-2]*x1[t-1] + e[3,t]
    end
    
    y1 = x1 .+ g .* generar_igarch(total_n)
    y2 = x2 .+ g .* generar_igarch(total_n)
    y3 = x3 .+ g .* generar_igarch(total_n)
    
    return y1[burn_in+1:end], y2[burn_in+1:end], y3[burn_in+1:end]
end

# ==========================================
# modulo 2: motor de metricas (optimizado)
# ==========================================

@inline function simbolizar!(s, x)
    @inbounds for i in 2:length(x)
        s[i-1] = x[i] > x[i-1] ? 1 : 0
    end
end

function calcular_pste(s_src, s_tar, s_cnd)
    n = length(s_tar) - 1
    c_f, c_c, c_xyz, c_yz = zeros(Int, 16), zeros(Int, 8), zeros(Int, 8), zeros(Int, 4)
    
    @inbounds for i in 1:n
        yf, yp, xp, zp = s_tar[i+1], s_tar[i], s_src[i], s_cnd[i]
        c_f[1 + yf + 2*yp + 4*zp + 8*xp] += 1
        c_c[1 + yf + 2*yp + 4*zp] += 1
        c_xyz[1 + xp + 2*yp + 4*zp] += 1
        c_yz[1 + yp + 2*zp] += 1
    end
    
    function h_func(counts)
        tot = sum(counts)
        tot == 0 && return 0.0
        h = 0.0
        for c in counts
            if c > 0
                p = c / tot
                h -= p * log2(p)
            end
        end
        return h
    end
    
    return h_func(c_c) + h_func(c_xyz) - h_func(c_f) - h_func(c_yz)
end

function calcular_cgci(src, tar, cnd, p=2)
    N = length(tar); Y = tar[p+1:N]; n_obs = length(Y)
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
# modulo 3: validacion estadistica
# ==========================================

function es_significativo_pste(s1, s2, s3; n_surr=40)
    val_real = calcular_pste(s1, s2, s3)
    s_shuf = copy(s1); cont = 0
    for _ in 1:n_surr
        shuffle!(s_shuf)
        calcular_pste(s_shuf, s2, s3) >= val_real && (cont += 1)
    end
    # formula rigurosa de montecarlo
    return ((cont + 1) / (n_surr + 1)) <= 0.05
end

function es_significativo_cgci(x1, x2, x3; n_surr=40)
    val_real = calcular_cgci(x1, x2, x3)
    x_shuf = copy(x1); cont = 0
    for _ in 1:n_surr
        shuffle!(x_shuf)
        calcular_cgci(x_shuf, x2, x3) >= val_real && (cont += 1)
    end
    # formula rigurosa de montecarlo
    return ((cont + 1) / (n_surr + 1)) <= 0.05
end

# ==========================================
# modulo 4: main
# ==========================================

function main()
    Random.seed!(123) # para que salgan igual las simulaxionmes
    n_realizaciones = 100
    tamanos = [512, 2048]
    
    # acumuladores de resultafos
    final_pste = zeros(Int, 2, 6)
    final_cgci = zeros(Int, 2, 6)

    println("\narrancando benchmarking sistema 8...")
    flush(stdout)

    for (idx, n) in enumerate(tamanos)
        println("=> procesando tamano n = $n:")
        flush(stdout)
        for r in 1:n_realizaciones
            # barra actualizable para ver el prpceso en vivo
            print("\r   realizacion $r de 100...") 
            flush(stdout)
            
            y1, y2, y3 = generar_datos_sistema8(n)
            
            s1, s2, s3 = zeros(Int, n), zeros(Int, n), zeros(Int, n)
            simbolizar!(s1, y1); simbolizar!(s2, y2); simbolizar!(s3, y3)
            
            es_significativo_pste(s1, s2, s3) && (final_pste[idx, 1] += 1) 
            es_significativo_pste(s2, s1, s3) && (final_pste[idx, 2] += 1) 
            es_significativo_pste(s2, s3, s1) && (final_pste[idx, 3] += 1) 
            es_significativo_pste(s3, s2, s1) && (final_pste[idx, 4] += 1) 
            es_significativo_pste(s1, s3, s2) && (final_pste[idx, 5] += 1) 
            es_significativo_pste(s3, s1, s2) && (final_pste[idx, 6] += 1) 

            es_significativo_cgci(y1, y2, y3) && (final_cgci[idx, 1] += 1)
            es_significativo_cgci(y2, y1, y3) && (final_cgci[idx, 2] += 1)
            es_significativo_cgci(y2, y3, y1) && (final_cgci[idx, 3] += 1)
            es_significativo_cgci(y3, y2, y1) && (final_cgci[idx, 4] += 1)
            es_significativo_cgci(y1, y3, y2) && (final_cgci[idx, 5] += 1)
            es_significativo_cgci(y3, y1, y2) && (final_cgci[idx, 6] += 1)
        end
        println(" ¡listo!")
        flush(stdout)
    end

    println("\ntabla de resultados sistema 8")
    println("-"^70)
    println(" n      | x1->x2 | x2->x1 | x2->x3 | x3->x2 | x1->x3 | x3->x1")
    println("-"^70)
    println("pste (m=2)")
    for (i, n) in enumerate(tamanos)
        @printf("n=%-5d |   %d    |   %d    |   %d    |   %d    |   %d    |   %d\n", 
                n, final_pste[i,1], final_pste[i,2], final_pste[i,3], final_pste[i,4], final_pste[i,5], final_pste[i,6])
    end
    println("-"^70)
    println("cgci (p=2)")
    for (i, n) in enumerate(tamanos)
        @printf("n=%-5d |   %d    |   %d    |   %d    |   %d    |   %d    |   %d\n", 
                n, final_cgci[i,1], final_cgci[i,2], final_cgci[i,3], final_cgci[i,4], final_cgci[i,5], final_cgci[i,6])
    end
end

main()