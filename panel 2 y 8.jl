using Plots
using Random

# Backend GR
gr()

# Semilla para reproducibilidad
Random.seed!(150) 

# Parámetros de simulación
n_transitorio = 1000 
n_grafica = 512
n_total = n_transitorio + n_grafica

# ==========================================
# 1. GENERACIÓN DE DATOS (MÉTODO 2 Y 8)
# ==========================================
x1, x2, x3 = zeros(n_total), zeros(n_total), zeros(n_total)
e = randn(3, n_total)

for t in 3:n_total
    x1[t] = 0.7 * x1[t-1] + e[1,t]
    x2[t] = 0.3 * x2[t-1] + 0.5 * x2[t-2] * x1[t-1] + e[2,t]
    x3[t] = 0.3 * x3[t-1] + 0.5 * x3[t-2] * x1[t-1] + e[3,t]
end

idx = n_transitorio+1:n_total
x1_f, x2_f, x3_f = x1[idx], x2[idx], x3[idx]

function igarch(n; a0=0.2, a1=0.9, b1=0.1)
    z, s2, err = zeros(n), zeros(n), randn(n)
    s2[1] = a0
    for t in 2:n
        s2[t] = a0 + a1 * z[t-1]^2 + b1 * s2[t-1]
        z[t] = sqrt(s2[t]) * err[t]
    end
    return z
end

g = 1.0 
y1_f = x1_f .+ g .* igarch(n_total)[idx]
y2_f = x2_f .+ g .* igarch(n_total)[idx]
y3_f = x3_f .+ g .* igarch(n_total)[idx]

# ==========================================
# 2. CONFIGURACIÓN DE NEGRO INTENSO (DARK BLACK)
# ==========================================
tiempo = 1:n_grafica

# Definimos las fuentes especificando color NEGRO (:black) y peso NEGRITA (:bold)
f_guia  = font(12, :bold, :black)   # Etiquetas X1, X2, X3
f_tick  = font(10, :bold, :black)   # Números de los ejes (parámetros)
f_title = font(13, :bold, :black)   # Títulos (a) y (b)

estilo_contraste = (
    legend = false,
    xlims = (0, 512),
    lw = 1.0,                       # Línea un poco más gruesa para que resalte
    guidefont = f_guia, 
    tickfont = f_tick,
    titlefont = f_title,
    grid = false,
    framestyle = :box,
    # ESTAS LÍNEAS FORZAN EL NEGRO OSCURO EN TODO EL GRÁFICO
    foreground_color_text = :black,   # Color de todas las letras
    foreground_color_axis = :black,   # Color de las marcas de los ejes
    foreground_color_border = :black, # Color del recuadro del gráfico
    foreground_color_guide = :black   # Color de los nombres de los ejes
)

c1, c2, c3 = :darkblue, :red, :limegreen

# ==========================================
# 3. RENDERIZADO DE PANELES
# ==========================================

# Sistema 2
p1 = plot(tiempo, x1_f; ylabel="X1", title="(a) System 2", color=c1, ylims=(-5,5), estilo_contraste...)
p2 = plot(tiempo, x2_f; ylabel="X2", color=c2, ylims=(-5,5), estilo_contraste...)
p3 = plot(tiempo, x3_f; ylabel="X3", xlabel="t", color=c3, ylims=(-10,20), estilo_contraste...)

# Sistema 8
p4 = plot(tiempo, y1_f; ylabel="X1", title="(b) System 8", color=c1, ylims=(-5,5), estilo_contraste...)
p5 = plot(tiempo, y2_f; ylabel="X2", color=c2, ylims=(-5,5), estilo_contraste...)
p6 = plot(tiempo, y3_f; ylabel="X3", xlabel="t", color=c3, ylims=(-10,20), estilo_contraste...)

plot_final = plot(
    p1, p4, p2, p5, p3, p6, 
    layout = (3, 2), 
    size = (1000, 800), 
    margin = 6Plots.mm,
    dpi = 300
)

display(plot_final)