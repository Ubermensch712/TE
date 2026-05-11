using Plots
using Random

# configuración del backend
gr()

# ==========================================
# bloque 1: generadores de datos
# ==========================================

# esta función genera la dinámica base del sistema 2
function generar_datos_sistema2(n_total)
    x1, x2, x3 = zeros(n_total), zeros(n_total), zeros(n_total)
    e = randn(3, n_total)

    for t in 3:n_total
        x1[t] = 0.7 * x1[t-1] + e[1,t]
        x2[t] = 0.3 * x2[t-1] + 0.5 * x2[t-2] * x1[t-1] + e[2,t]
        x3[t] = 0.3 * x3[t-1] + 0.5 * x3[t-2] * x1[t-1] + e[3,t]
    end
    return x1, x2, x3
end

# esta función genera el ruido igarch para el sistema 8
function calcular_igarch(n; a0=0.2, a1=0.9, b1=0.1)
    z, s2, err = zeros(n), zeros(n), randn(n)
    s2[1] = a0
    for t in 2:n
        s2[t] = a0 + a1 * z[t-1]^2 + b1 * s2[t-1]
        z[t] = sqrt(s2[t]) * err[t]
    end
    return z
end

# ==========================================
# bloque 2: configuración visual
# ==========================================

# define los estilos de fuente y el diccionario de contraste
function obtener_estilo_visual()
    # fuentes en negrita y color negro
    f_guia  = font(12, :bold, :black)
    f_tick  = font(10, :bold, :black)
    f_title = font(13, :bold, :black)

    estilo = (
        legend = false,
        xlims = (0, 512),
        lw = 1.0,
        guidefont = f_guia, 
        tickfont = f_tick,
        titlefont = f_title,
        grid = false,
        framestyle = :box,
        foreground_color_text = :black,
        foreground_color_axis = :black,
        foreground_color_border = :black,
        foreground_color_guide = :black
    )
    return estilo
end

# ==========================================
# bloque 3: ejecución principal (main)
# ==========================================

function main()
    # inicialización de parámetros
    Random.seed!(150) 
    n_transitorio = 1000 
    n_grafica = 512
    n_total = n_transitorio + n_grafica
    idx = n_transitorio+1:n_total
    tiempo = 1:n_grafica
    g = 1.0 

    # generación de datos del sistema 2
    x1, x2, x3 = generar_datos_sistema2(n_total)
    
    # limpieza del periodo transitorio para sistema 2
    x1_f, x2_f, x3_f = x1[idx], x2[idx], x3[idx]

    # generación de datos del sistema 8 (superposición)
    # nota: se llama a la función igarch individualmente para cada serie
    y1_f = x1_f .+ g .* calcular_igarch(n_total)[idx]
    y2_f = x2_f .+ g .* calcular_igarch(n_total)[idx]
    y3_f = x3_f .+ g .* calcular_igarch(n_total)[idx]

    # obtención de parámetros estéticos
    estilo = obtener_estilo_visual()
    c1, c2, c3 = :darkblue, :red, :limegreen

    # renderizado de paneles del sistema 2
    p1 = plot(tiempo, x1_f; ylabel="X1", title="(a) System 2", color=c1, ylims=(-5,5), estilo...)
    p2 = plot(tiempo, x2_f; ylabel="X2", color=c2, ylims=(-5,5), estilo...)
    p3 = plot(tiempo, x3_f; ylabel="X3", xlabel="t", color=c3, ylims=(-10,20), estilo...)

    # renderizado de paneles del sistema 8
    p4 = plot(tiempo, y1_f; ylabel="X1", title="(b) System 8", color=c1, ylims=(-5,5), estilo...)
    p5 = plot(tiempo, y2_f; ylabel="X2", color=c2, ylims=(-5,5), estilo...)
    p6 = plot(tiempo, y3_f; ylabel="X3", xlabel="t", color=c3, ylims=(-10,20), estilo...)

    # compilación del gráfico final
    plot_final = plot(
        p1, p4, p2, p5, p3, p6, 
        layout = (3, 2), 
        size = (1000, 800), 
        margin = 6Plots.mm,
        dpi = 300
    )

    return plot_final
end

# ejecución y despliegue
figura = main()
display(figura)
