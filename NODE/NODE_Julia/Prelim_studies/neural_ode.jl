using DifferentialEquations
using Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  a, b, c, d = p
  du[1] = dx = a*x - b*x*y
  du[2] = dy = -c*y + d*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra, u0, tspan, p)
sol = solve(prob, Tsit5())

# Plot the solution
plot(sol)
savefig("LV_ode.png")

function loss(p)
  sol = solve(prob, Tsit5(), p=p, saveat = tsteps)
  loss = sum(abs2, sol.-1)
  return loss, sol
end

callback = function (p, l, pred)
  display(l)
  plt = plot(pred, ylim = (0, 6))
  display(plt)
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

result_ode = DiffEqFlux.sciml_train(loss, p,
                                    cb = callback,
                                    maxiters = 100)


prob_final = ODEProblem(lotka_volterra, u0, tspan, result_ode)
sol_final = solve(prob_final,Tsit5())
plot(sol_final)
savefig("LV_ode_optimized.png") # Both state components close to 1

# Start julia interpreter and type the following
# include("neural_ode.jl") # This will take a while
# After finishing
# Now you can change u0, tspan, p and re-run
# prob = ODEProblem(lotka_volterra,u0,tspan,p)
# sol=solve(prob);plot(sol); # This will be superfast (JIT)