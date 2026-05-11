using Random, Statistics, LinearAlgebra, Printf

# ==========================================
# bloque 1: generadores de datos
# ==========================================

# esta función genera la dinámica del sistema 2 (estacionario)
function generar_datos_sistema2(n; burn_in=1000)
    total_n = n + burn_in
    x1, x2, x3 = zeros(total_n), zeros(total_n), zeros(total_n)
    
    # generación de ruido independiente para cada variable
    e1 = randn(total_n)
    e2 = randn(total_n)
    e3 = randn(total_n)
    
    for t in 3:total_n
        x1[t] = 0.7 * x1[t-1] + e1[t]
        x2[t] = 0.3 * x2[t-1] + 0.5 * x2[t-2] * x1[t-1] + e2[t]
        x3[t] = 0.3 * x3[t-1] + 0.5 * x3[t-2] * x1[t-1] + e3[t]
    end
    
    # eliminación del periodo transitorio
    return x1[burn_in+1:end], x2[burn_in+1:end], x3[burn_in+1:end]
end

# ==========================================
# bloque 2: motores de cálculo de información
# ==========================================

# simbolización de bandt-pompe de orden m=2
@inline function simbolizar!(s, x)
    @inbounds for i in 2:length(x)
        s[i-1] = x[i] > x[i-1] ? 1 : 0
    end
end

# cálculo de pste mediante mapeo binario indexado de 16 estados
function calcular_pste(s_src, s_tar, s_cnd)
    n = length(s_tar) - 1
    # vectores de frecuencia para las distribuciones conjuntas
    c_f, c_c, c_xyz, c_yz = zeros(Int, 16), zeros(Int, 8), zeros(Int, 8), zeros(Int, 4)
    
    @inbounds for i in 1:n
        yf, yp, xp, zp = s_tar[i+1], s_tar[i], s_src[i], s_cnd[i]
        # mapeo de estados para optimizar el conteo
        c_f[1 + yf + 2*yp + 4*zp + 8*xp] += 1
        c_c[1 + yf + 2*yp + 4*zp] += 1
        c_xyz[1 + xp + 2*yp + 4*zp] += 1
        c_yz[1 + yp + 2*zp] += 1
    end
    
    # cálculo de entropía de shannon optimizado para memoria
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

# cálculo del índice de causalidad de granger condicional (cgci)
function calcular_cgci(src, tar, cnd, p=2)
    n = length(tar); y = tar[p+1:n]; n_obs = length(y)
    
    # construcción del modelo restringido (pasado del objetivo y condicional)
    x_r = ones(n_obs, 1 + 2*p)
    for i in 1:p
        x_r[:, 1+i] = tar[p+1-i : n-i]
        x_r[:, 1+p+i] = cnd[p+1-i : n-i]
    end
    
    # construcción del modelo completo (incluye pasado de la fuente)
    x_f = hcat(x_r, zeros(n_obs, p))
    for i in 1:p
        x_f[:, 1+2*p+i] = src[p+1-i : n-i]
    end
    
    # resolución por mínimos cuadrados y cálculo de varianzas
    var_r = var(y - x_r * (x_r \ y))
    var_f = var(y - x_f * (x_f \ y))
    
    return (var_f > 0 && var_r > var_f) ? log(var_r / var_f) : 0.0
end

# ==========================================
# bloque 3: validación estadística (shuffling)
# ==========================================

function es_significativo_pste(s_src, s_tar, s_cnd; n_surr=40)
    val_real = calcular_pste(s_src, s_tar, s_cnd)
    s_shuf = copy(s_src); cont = 0
    for _ in 1:n_surr
        shuffle!(s_shuf)
        calcular_pste(s_shuf, s_tar, s_cnd) >= val_real && (cont += 1)
    end
    return (cont / n_surr) < 0.05
end

function es_significativo_cgci(src, tar, cnd; n_surr=40)
    val_real = calcular_cgci(src, tar, cnd)
    src_shuf = copy(src); cont = 0
    for _ in 1:n_surr
        shuffle!(src_shuf)
        calcular_cgci(src_shuf, tar, cnd) >= val_real && (cont += 1)
    end
    return (cont / n_surr) < 0.05
end

# ==========================================
# bloque 4: controlador de benchmarking (main)
# ==========================================

function main()
    Random.seed!(123)
    n_realizaciones = 100
    tamanos_n = [512, 2048]
    
    # contenedores para las tablas de resultados
    final_pste = zeros(Int, 2, 6)
    final_cgci = zeros(Int, 2, 6)

    println("\niniciando benchmarking sistema 2 (papana 2017)...")

    for (idx, n) in enumerate(tamanos_n)
        println("=> procesando tamano n = $n:")
        for r in 1:n_realizaciones
            # indicador de progreso por realización
            print("\r   simulacion $r de 100...")
            
            # 1. generación de los datos originales
            x1, x2, x3 = generar_datos_sistema2(n)
            
            # 2. transformación simbólica
            s1, s2, s3 = zeros(Int, n), zeros(Int, n), zeros(Int, n)
            simbolizar!(s1, x1); simbolizar!(s2, x2); simbolizar!(s3, x3)
            
            # 3. evaluación de las 6 direcciones posibles para pste
            es_significativo_pste(s1, s2, s3) && (final_pste[idx, 1] += 1) # x1->x2
            es_significativo_pste(s2, s1, s3) && (final_pste[idx, 2] += 1) # x2->x1
            es_significativo_pste(s2, s3, s1) && (final_pste[idx, 3] += 1) # x2->x3
            es_significativo_pste(s3, s2, s1) && (final_pste[idx, 4] += 1) # x3->x2
            es_significativo_pste(s1, s3, s2) && (final_pste[idx, 5] += 1) # x1->x3
            es_significativo_pste(s3, s1, s2) && (final_pste[idx, 6] += 1) # x3->x1

            # 4. evaluación de las 6 direcciones para cgci
            es_significativo_cgci(x1, x2, x3) && (final_cgci[idx, 1] += 1)
            es_significativo_cgci(x2, x1, x3) && (final_cgci[idx, 2] += 1)
            es_significativo_cgci(x2, x3, x1) && (final_cgci[idx, 3] += 1)
            es_significativo_cgci(x3, x2, x1) && (final_cgci[idx, 4] += 1)
            es_significativo_cgci(x1, x3, x2) && (final_cgci[idx, 5] += 1)
            es_significativo_cgci(x3, x1, x2) && (final_cgci[idx, 6] += 1)
        end
        println(" listo.")
    end

    # impresión final de los resultados en formato tabla
    println("\ntablas de exito (porcentaje de rechazo h0)")
    println("-"^75)
    println(" n      | x1->x2 | x2->x1 | x2->x3 | x3->x2 | x1->x3 | x3->x1")
    println("-"^75)
    
    println("pste (m=2)")
    for (i, n) in enumerate(tamanos_n)
        @printf("n=%-5d |   %d   |   %d   |   %d   |   %d   |   %d   |   %d\n", 
                n, final_pste[i,1], final_pste[i,2], final_pste[i,3], final_pste[i,4], final_pste[i,5], final_pste[i,6])
    end
    
    println("\ncgci (p=2)")
    for (i, n) in enumerate(tamanos_n)
        @printf("n=%-5d |   %d   |   %d   |   %d   |   %d   |   %d   |   %d\n", 
                n, final_cgci[i,1], final_cgci[i,2], final_cgci[i,3], final_cgci[i,4], final_cgci[i,5], final_cgci[i,6])
    end
    println("-"^75)
end

# ejecutar el benchmarking
main()
