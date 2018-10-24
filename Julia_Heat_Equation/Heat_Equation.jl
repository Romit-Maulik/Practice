#=
This is a Julia script for a simple FTCS heat equation
Romit Maulik - CFD Laboratory
=#

#For plotting
using PyPlot

#Initialize array with ICs
function init(nstart, nend, sol, x)
	i = nstart
	pi = 3.14159
	while i <= nend
		x[i] = (i-2)*dx #x starts from zero and goes to nend-1 (periodic conditions) - x[2] = 0.0, x[130]= x[2], x[1] = x[129]
		sol[i] = sin.(2.0*pi*x[i])
		i = i + 1
	end

	x[end] = x[2]
	x[1] = x[end-1]

	sol[end] = sol[2]
	sol[1] = sol[end-1]

	return sol
end


function euler_t_step(alpha,dx,dt,nstart,nend,sol)

	i = nstart
	#Make temporary copy of array
	temp_sol = copy(sol)

	while i <= nend
		temp_sol[i] = sol[i] + alpha*dt/(dx^2)*(sol[i+1]+sol[i-1]-2.0*sol[i])
		i = i + 1
	end

	temp_sol[end] = temp_sol[2]
	temp_sol[1] = temp_sol[end-1]

	i = 1
	while i <= length(sol)
		sol[i] = temp_sol[i]
		i = i+1
	end
end


function solve_ftcs(alpha,dt,dx,sol,nstart,nend,final_time)

t = 0.0

	while t < final_time
		t = t+dt
		println("Time is: ",t)

		euler_t_step(alpha,dx,dt,nstart,nend,sol)

	end

	return sol

end

function plot_solution(x,sol,string_val)

plot(x[2:end-1],sol[2:end-1],linewidth=1.0,label=string_val)

end

#FTCS for a given number of time steps

alpha = 0.2# Float64
npoints = 130# Int64 - 128 + 2 ghost points

nstart = 2
nend = 129

lx = 1.0# Float64

dx = lx/(nend-nstart+1)#Automatically Float64

#Create array of zeros to store our solution
sol = zeros(Float64,npoints)
x = zeros(Float64,npoints)

sol = init(nstart, nend, sol, x)

#println(sol)
#println(x)

dt = 0.4*(dx^2)/(alpha)
final_time = 0.2

figure()
xlabel("x")
ylabel("Temperature")

plot_solution(x,sol,"Initial time")

sol = solve_ftcs(alpha,dt,dx,sol,nstart,nend,final_time)

plot_solution(x,sol,"t=0.2")
legend()
#show()

savefig("Test.png")
