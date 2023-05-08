### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 849dbc25-8dd1-416b-b706-672e826eec6a
using CSV, Plots, GLM, LsqFit, Statistics,StatsPlots,HypothesisTests,Symbolics

# ╔═╡ faef2569-8983-4704-9e1a-7080fb16dd06
md"
=====================================================================================
#### ISLR2_2.1 What Is Statistical Learning ?
##### file: PCM20230424\_ISLR2\_2.1\_What\_Is\_Statistical\_Learning.jl
##### Julia/Pluto (1.8.5/0.19.14) by PCM *** 2023/05/08 ***
=====================================================================================
"

# ╔═╡ 622d27ff-e36f-4b69-949d-1fa78a8413ce
md"
---
#### 1. The *Advertising* Data Set
##### 1.1 Dataframe
"

# ╔═╡ 7b047630-b637-47ee-9aee-6d685dd3e773
advertisingDataFrame = CSV.File("C:/Users/claus/Nextcloud/PCM_Book/PCM2023_ISLR_Data/AdvertisingData/AdvertisingData.csv")

# ╔═╡ f698b886-e349-436f-932d-657bf7ff1eb5
md"
---
##### 1.2  2D-Plot
"

# ╔═╡ de0c31cf-a954-41fb-afca-a03cf28364dc
let
	Plots.plot(title="TV-Advertising -> Sales (ISLR2e, Fig. 2.1 - left)")
	Plots.plot!(advertisingDataFrame.TV, advertisingDataFrame.Sales, seriestype=:scatter, label="(x,y)", xlabel="advertisingDataFrame.TV", ylabel="advertisingDataFrame.Sales")
end # let

# ╔═╡ b852829d-0da2-44db-b7ed-9bf390858882
md"
---
##### 1.3 Linear Regression Models
"

# ╔═╡ 009a4323-0593-4900-b0d9-e2c4fc7a8756
let
	ols_lin = lm(@formula(Sales ~ TV), advertisingDataFrame)
	yHat = predict(ols_lin)
	Plots.plot(title="TV-Advertising -> Sales (ISLR2e, Fig. 2.1 - left)")
	Plots.plot!(advertisingDataFrame.TV, advertisingDataFrame.Sales, seriestype=:scatter, label="(x,y)", xlabel="advertisingDataFrame.TV", ylabel="advertisingDataFrame.Sales")
	Plots.plot!(advertisingDataFrame.TV, yHat, seriestype=:line, width=2, label="(x,y-hat)")
end # let

# ╔═╡ 4d9fc2b5-3ee4-4acf-be66-d2881f9f994a
let
	ols_lin = lm(@formula(Sales ~ Radio), advertisingDataFrame)
	yHat = predict(ols_lin)
	Plots.plot(title="Radio-Advertising -> Sales (ISLR2e, Fig. 2.1 - middle)")
	Plots.plot!(advertisingDataFrame.Radio, advertisingDataFrame.Sales, seriestype=:scatter, label="(x,y)", xlabel="advertisingDataFrame.Radio", ylabel="advertisingDataFrame.Sales")
	Plots.plot!(advertisingDataFrame.Radio, yHat, seriestype=:line, width=2, label="(x,y-hat)")
end # let

# ╔═╡ da5a6876-ef9f-4d73-bd3b-1369c70bc103
let
	ols_lin = lm(@formula(Sales ~ Newspaper), advertisingDataFrame)
	yHat = predict(ols_lin)
	Plots.plot(title="Newsp-Advertising -> Sales (ISLR2e, Fig. 2.1 - right)")
	Plots.plot!(advertisingDataFrame.Newspaper, advertisingDataFrame.Sales, seriestype=:scatter, label="(x,y)", xlabel="advertisingDataFrame.Newspaper", ylabel="advertisingDataFrame.Sales")
	Plots.plot!(advertisingDataFrame.Newspaper, yHat, seriestype=:line, width=2, label="(x,y-hat)")
end # let

# ╔═╡ 24576004-b1d2-43e1-a861-eba45d1fda0e
md"
---
#### 2. The *Income1* Data Set

Both *Income* data sets are generated from numerical simulations. The *true* model function $f(Education, Seniority)$ is depicted in Figure 2.3 (ISLR2, 2021, p.18) as a *blue* surface. Its mathematical form is *not* published by the authors. So we have to estimate $f$ by various approaches.
"

# ╔═╡ 1feee341-401b-4afd-8970-7c2804fc3f89
md"
---
##### 2.1 Dataframe
"

# ╔═╡ 8f6de19f-c4ff-46dd-a120-269ebf647578
income1DataFrame = CSV.File("C:/Users/claus/Nextcloud/PCM_Book/PCM2023_ISLR_Data/IncomeData/Income1Data.csv")

# ╔═╡ d800b01d-477b-4776-bd2c-848fff8a5da9
md"
---
##### 2.2 ISLR2, Fig.2.2
###### 2.2.1 2D-Plot
"

# ╔═╡ 7ad28ec0-74ad-4f15-8235-259f091622a6
let
	xdata = income1DataFrame.Education
	ydata = income1DataFrame.Income
	Plots.plot(title="Income -> Education (ISLR2e, Fig. 2.2 - left)", xlims=(9,23), ylims=(5,95))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income1DataFrame.Education", ylabel="income1DataFrame.Income")
end # let

# ╔═╡ 55a0b4d7-890f-4d32-a10f-002631906bda
md"
---
###### 2.2.2 Linear Regression Model $f(X)=\beta_0 + \beta_1X_1$
"

# ╔═╡ bfcdfaf5-50c1-452a-946b-941e5c4c1041
let
	xdata = income1DataFrame.Education
	ydata = income1DataFrame.Income
	#--------------------------------------------------------------------------------
	ols_lin = lm(@formula(Income ~ 1 + Education), income1DataFrame)
	yhat = predict(ols_lin)
	e = residuals(ols_lin)
	sse = round(e'e, digits=2)
	#--------------------------------------------------------------------------------
	Plots.plot(title="Income -> Education (ISLR2e, Fig. 2.2 - (new))", xlims=(9,23), ylims=(5,95))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income1DataFrame.Education", ylabel="income1DataFrame.Income")
	Plots.plot!(xdata, yhat, seriestype=:line, width=2, label="(x,yhat)")
	Plots.plot!(map(((x, y, yh) -> ([x, x],[y, yh])), xdata, ydata, yhat), label="")
	#--------------------------------------------------------------------------------
	annotate!((19.2,22,"SSE = $sse ( = error sum of squares)", 10))
end # let

# ╔═╡ 5479bd68-aafa-47c6-bce5-d5a8817c131a
md"
---
###### 2.2.3 Nonlinear Regression Model using the [Generalized Logistic Function](https://en.wikipedia.org/wiki/Generalised_logistic_function)

$f(x) = L + (U-L) \cdot \frac{1}{1+e^{-k\left(x - x_{k_{max}}\right)}}$

where:
- **$k_{max}$** = $max(k)$ = max. growth rate *k* of $X$
- **$U$** = Upper (right) asymptote of $X$
- **$L$** = Lower (left) asymptote of $X$
- **$x_{k_{max}}$** = $\underset{X}{\operatorname{argmax}\;k} \;\; \text{(= }x\text{ is the inflection point of } k)$ 

The above function $f(X)$ is a solution of *differential equation* describing a [*(logistic) growth process*](https://en.wikipedia.org/wiki/Logistic_function). We derive the differential equation for the simple logistic function $(0 < f_{simple}(X) < 1)$ in the appendix:

$f_{simple}(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}.$ 

"

# ╔═╡ 92ea0bb5-d318-4758-83c9-ac5960918e12
let
	#--------------------------------------------------------------------------------
	xdata = income1DataFrame.Education
	ydata = income1DataFrame.Income
	#--------------------------------------------------------------------------------
	myModel(x, p) = p[1] .+ (p[2]-p[1]) .* (1 ./ ((1 .+ exp.(-p[3] .* (x .- p[4])))))
	parms0 = [18.0, 78.0, 0.5, 16.0]                    # initial estimates of parms
	myModel(xdata, parms0)
	fit = curve_fit(myModel, xdata, ydata, parms0)
	e = fit.resid
	sse = round(e'e, digits=2)
	parms_hat = coef(fit)
	L  = round(parms_hat[1], digits=2)                    # greatest Lower bound (gLb)
	U  = round(parms_hat[2], digits=2)                    # least Upper bound (lUb)
	k  = round(parms_hat[3], digits=2)                    # max. growth rate k
	x_k_max = round(parms_hat[4], digits=2)               # point x of k_max
	yhat = myModel(xdata, parms_hat)
	#--------------------------------------------------------------------------------
	Plots.plot(title="Income -> Education (ISLR2e, Fig. 2.2 - right)", xlims=(9,23), ylims=(5,95))
	Plots.plot!(xdata, ydata, seriestype=:scatter, xlabel="income1DataFrame.Education", ylabel="income1DataFrame.Income", label="(x,y)")
	Plots.plot!(xdata, yhat, seriestype=:line, label="(x,yhat)", w=2)
	Plots.plot!(map(((x, y, yh) -> ([x, x],[y, yh])), xdata, ydata, yhat), label="")
	#--------------------------------------------------------------------------------
	annotate!((21.4,46,"x_k_max = $x_k_max", 10))
	annotate!((20.3,40,"k=$k (= max. growth rate)", 10))
	annotate!((19.4,34,"U=$U (= Upper (right) asymptote)", 10))
	annotate!((19.6,28,"L=$L (= Lower (left) asymptote)", 10))
	annotate!((19.2,22,"SSE = $sse ( = error sum of squares)", 10))
	#--------------------------------------------------------------------------------
end # let

# ╔═╡ e5a0b52d-f482-43fa-9efc-84ecd0711794
md"""

The derivative $f'(x)$ is the description of the logistic growth process and is called [*logistic differential equation*](https://en.wikipedia.org/wiki/Logistic_function) with boundary condition $f(0)= \frac{1}{2}$.


This can be simplified to the product of two terms:


$$f'(x) = \frac{e^x}{(e^x+1)^2} = \frac{e^x}{(e^x+1)(e^x+1)} = \frac{e^x}{(e^x+1)} \cdot \frac{1}{(e^x+1)} = f(x)(1-f(x)).$$


"""

# ╔═╡ 5adf8b34-2d41-4b15-8c13-0cbf84ed0cce
md"
---
#### 3. The *Income2* Data Set
##### 3.1 Dataframe

"

# ╔═╡ ec30ce62-3699-40f8-88c3-8e394d6d826e
income2DataFrame = CSV.File("C:/Users/claus/Nextcloud/PCM_Book/PCM2023_ISLR_Data/IncomeData/Income2Data.csv")

# ╔═╡ c54e5513-f30a-4bc1-afe0-102ba43a6f87
md"
---
##### 3.2 ISLR2, Fig.2.3 $Income \rightarrow Education$
###### 3.2.1 2D-Plot
" 

# ╔═╡ a559d47a-bdf9-4229-b363-d33ff4994de6
let
	xdata = income2DataFrame.Education
	ydata = income2DataFrame.Income
	Plots.plot(title="Income -> Education (ISLR2e, Fig. 2.3 - (new))", xlims=(9,22), ylims=(5,120))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income2DataFrame.Education", ylabel="income2DataFrame.Income")
end # let

# ╔═╡ a0e54499-eb79-4381-81eb-acfd8826511e
md"
---
###### 3.2.2 Linear Regression Model $f(X)=\beta_0 + \beta_1X_1$
"

# ╔═╡ 314821cf-1739-454e-a6eb-75b3bed752b3
let
	xdata = income2DataFrame.Education
	ydata = income2DataFrame.Income
	#--------------------------------------------------------------------------------
	ols_lin = lm(@formula(Income ~ 1 + Education), income2DataFrame)
	yhat = predict(ols_lin)
	e    = residuals(ols_lin)
	sse  = round(e'e, digits=2)
	#--------------------------------------------------------------------------------
	Plots.plot(title="Linear OLS: Income -> Education (ISLR2e, cf. Fig.2.3)", xlims=(9,22), ylims=(5,120))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income2DataFrame.Education", ylabel="income2DataFrame.Income")
	Plots.plot!(xdata, yhat, seriestype=:line, width=2, label="(x,yhat)")
	Plots.plot!(map(((x, y, yh) -> ([x, x],[y, yh])), xdata, ydata, yhat), label="")
	#--------------------------------------------------------------------------------
	annotate!((18.4,22,"SSE = $sse ( = error sum of squares)", 10))
	#--------------------------------------------------------------------------------
end # let

# ╔═╡ d1982304-7008-4486-a16d-5d4064935ebc
md"
---
###### 3.2.3 Nonlinear Regression Models using the [Generalized Logistic Function](https://en.wikipedia.org/wiki/Generalised_logistic_function)

$f(x) = L + (U-L) \cdot \frac{1}{1+e^{-k\left(x - x_{k_{max}}\right)}}$

where:
- **$k_{max}$** = $max(k)$ = max. growth rate *k* of $X$
- **$U$** = Upper (right) asymptote of $X$
- **$L$** = Lower (left) asymptote of $X$
- **$x_{k_{max}}$** = $\underset{X}{\operatorname{argmax}\;k} \;\; \text{(= }x\text{ is the inflection point of rate } k)$ 

$\;$

The above function $f(X)$ is a solution of *differential equation* describing a [*(logistic) growth process*](https://en.wikipedia.org/wiki/Logistic_function). We derive the differential equation for the *simple* logistic function $(0 < f_{simple}(x) < 1)$ in the appendix:

$f_{simple}(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}.$ 

$\;$

$\;$



"

# ╔═╡ 09af8231-726f-4439-b123-a14170b5170e
let
	xdata = income2DataFrame.Education
	ydata = income2DataFrame.Income
	#--------------------------------------------------------------------------------
	myModel(x, p) = p[1] .+ (p[2]-p[1]) .* (1 ./ ((1 .+ exp.(-p[3] .* (x .- p[4])))))
	parms0 = [28.0, 88.0, 0.9, 16.0]                    # initial estimates of parms
	myModel(xdata, parms0)
	fit = curve_fit(myModel, xdata, ydata, parms0)
	e = fit.resid
	sse = round(e'e, digits=2)
	parms_hat = coef(fit)
	L  = round(parms_hat[1], digits=2)                    # greatest Lower bound (gLb)
	U  = round(parms_hat[2], digits=2)                    # least Upper bound (lUb)
	k  = round(parms_hat[3], digits=2)                    # max. growth rate k
	x_k_max = round(parms_hat[4], digits=2)               # point x of k_max
	yhat = myModel(xdata, parms_hat)
	#--------------------------------------------------------------------------------
	Plots.plot(title="Logistic OLS: Income -> Education (ISLR2e, cf.Fig.2.3)", xlims=(9,22), ylims=(5,120))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income2DataFrame.Education", ylabel="income2DataFrame.Income")
	Plots.plot!(xdata, yhat, seriestype=:line, label="(x,yhat)", w=2)
	Plots.plot!(map(((x, y, yh) -> ([x, x],[y, yh])), xdata, ydata, yhat), label="")
	#--------------------------------------------------------------------------------
	annotate!((20.55,46,"x_k_max = $x_k_max", 10))
	annotate!((19.5,40,"k=$k (= max. growth rate)", 10))
	annotate!((18.7,34,"U=$U (= Upper (right) asymptote)", 10))
	annotate!((19.0,28,"L=$L (= Lower (left) asymptote)", 10))
	annotate!((18.4,22,"SSE = $sse ( = error sum of squares)", 10))
	#--------------------------------------------------------------------------------
end # let

# ╔═╡ df58736c-ebd1-4020-915c-b32291985f3f
md"
---
##### 3.3 ISLR2, Fig.2.3 $Income \rightarrow Seniority$
###### 3.3.1 2D-Plot
"

# ╔═╡ 8e78b3e0-e97e-4b8d-9342-a180b26bbf03
let
	xdata = income2DataFrame.Seniority
	ydata = income2DataFrame.Income
	#--------------------------------------------------------------------------------
	Plots.plot(title="Income -> Seniority (ISLR2e, Fig. 2.3 - (new))", xlims=(9,200), ylims=(5,125))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income2DataFrame.Seniority", ylabel="income2DataFrame.Income")
	#--------------------------------------------------------------------------------
end # let

# ╔═╡ 2771a9f1-353f-4bfa-b4d9-a0deda6e031a
md"
---
###### 3.3.2 Linear Regression Model $f(X)=\beta_0 + \beta_1X_1$
"

# ╔═╡ 84c06bd1-4dbf-411f-96c4-97554146ca21
let
	xdata = income2DataFrame.Seniority
	ydata = income2DataFrame.Income
	#--------------------------------------------------------------------------------
	ols_lin = lm(@formula(Income ~ 1 + Seniority), income2DataFrame)
	yhat = predict(ols_lin)
	e    = residuals(ols_lin)
	sse  = round(e'e, digits=2)
	#--------------------------------------------------------------------------------
	Plots.plot(title="Linear OLS: Income -> Seniority (ISLR2e, cf. Fig.2.3)", xlims=(9,200), ylims=(5,125))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income2DataFrame.Seniority", ylabel="income2DataFrame.Income")
	Plots.plot!(xdata, yhat, seriestype=:line, width=2, label="(x,yhat)")
	Plots.plot!(map(((x, y, yh) -> ([x, x],[y, yh])), xdata, ydata, yhat), label="")
	#--------------------------------------------------------------------------------
	annotate!((140,15,"SSE = $sse ( = error sum of squares)", 10))
end # let

# ╔═╡ baa67da7-d394-4a1d-9eee-3a982c0f74c2
md"
---
###### 3.3.3 Nonlinear Regression Model using the [Generalized Logistic Function](https://en.wikipedia.org/wiki/Generalised_logistic_function)

$f(x) = L + \frac{U-L}{1+e^{-k\left(x - x_{k_{max}}\right)}}$

where:
- **$k_{max}$** = $max(k)$ = max. growth rate *k* of $X$
- **$U$** = Upper (right) asymptote of $X$
- **$L$** = Lower (left) asymptote of $X$
- **$x_{k_{max}}$** = $\underset{X}{\operatorname{argmax}\;k} \;\; \text{(= }x\text{ is the inflection point of rate } k)$ 

$\;$

The above function $f(x)$ is a solution of *differential equation* describing a [*(logistic) growth process*](https://en.wikipedia.org/wiki/Logistic_function). We derive the differential equation for the *simple* logistic function $f_{simple}(x)$ in the appendix:

$f_{simple}(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}.$ 

$\;$

$\;$
"

# ╔═╡ f3025a7a-0256-46c7-b296-4cc3a36c5372
let
	xdata = income2DataFrame.Seniority
	ydata = income2DataFrame.Income
	#------------------------------------------------------------------------------
	myModel(x, p) = p[1] .+ (p[2] .-p[1]) ./((1 .+ exp.(-p[3] .*(x .- p[4]))))
	parms0 = [40.0, 78.0, 4.0, 70.0]                      # initial estimates of parms
	myModel(xdata, parms0)
	fit = curve_fit(myModel, xdata, ydata, parms0)
	e = fit.resid
	sse = round(e'e, digits=2)
	parms_hat = coef(fit)
	L  = round(parms_hat[1], digits=2)                    # greatest Lower bound (gLb)
	U  = round(parms_hat[2], digits=2)                    # least Upper bound (lUb)
	k  = round(parms_hat[3], digits=2)                    # max. growth rate k
	x_k_max = round(parms_hat[4], digits=2)               # point x of k_max
	yhat = myModel(xdata, parms_hat)
	#------------------------------------------------------------------------------
	Plots.plot(title="Logistic OLS: Income -> Seniority (ISLR2e, cf. Fig.2.3)", xlims=(9,200), ylims=(5,125))
	Plots.plot!(xdata, ydata, seriestype=:scatter, label="(x,y)", xlabel="income2DataFrame.Seniority", ylabel="income2DataFrame.Income")
	Plots.plot!(xdata, yhat, seriestype=:line, label="(x,yhat)", w=2)
	Plots.plot!(map(((x, y, yh) -> ([x, x],[y, yh])), xdata, ydata, yhat), label="")
	#------------------------------------------------------------------------------
	annotate!((181.0,36,"x_k_max = $x_k_max", 10))
	annotate!((166,30,"k=$k (= max. growth rate)", 10))
	annotate!((152.7,24,"U=$U (= Upper (right) asymptote)", 10))
	annotate!((155.0,18,"L=$L (= Lower (left) asymptote)", 10))
	annotate!((146.4,12,"SSE = $sse ( = error sum of squares)", 10))
	#------------------------------------------------------------------------------
end # let

# ╔═╡ e31097bf-1cbf-4146-91b3-841b91aa11b3
md"
---
##### 3.4 ISLR2, Fig.2.3 $Education \times Seniority$
###### 3.4.1 2D-Plot
"

# ╔═╡ f98e88e4-6abf-41c6-86c4-60f61523e1bb
let
	#-------------------------------------------------------------------------------
	xs = income2DataFrame.Education
	ys = income2DataFrame.Seniority
	#-------------------------------------------------------------------------------
	plot(xs, ys, title="Income2 Data: Split at Inflection Points of Rate k", seriestype=:scatter, legend=:none, xlabel="Education", ylabel="Seniority")
	plot!([(16.0, 0),  (16, 220)])                # vertical line at x_k_max = 16
	plot!([( 9.5, 70), (22,  70)])                # horizontal line at y_k_max = 70
	#-------------------------------------------------------------------------------
	annotate!(20.7,  60, "y_k_max=70", 10)
	annotate!(14.7, 210, "x_k_max=16", 10)
	#-------------------------------------------------------------------------------
end # let

# ╔═╡ 1e5f39c1-b6e0-4088-b0dd-5e6162602bed
md"
---
###### 3.4.2 Group-based Frequencies and Means
"

# ╔═╡ 4da9634c-41ab-469f-a25b-dc618e65eacb
let
	xs  = income2DataFrame.Education
	ys  = income2DataFrame.Seniority 
	zs  = income2DataFrame.Income
	#-------------------------------------------------------------------
	lx_ly =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x <= 16.0 && y <= 70.0 ? z : 0, xs, ys, zs)))
	mean_lx_ly   = mean(lx_ly)
	length_lx_ly = length(lx_ly)
	#---------------------------------------------------------
	lx_hy =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x <= 16.0 && y > 70.0 ? z : 0, xs, ys, zs)))
	mean_lx_hy   = mean(lx_hy)
	length_lx_hy = length(lx_hy)
	#---------------------------------------------------------
	hx_ly =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x > 16.0 && y <= 70.0 ? z : 0, xs, ys, zs)))
	mean_hx_ly   = mean(hx_ly)
	length_hx_ly = length(hx_ly)
	#---------------------------------------------------------
	hx_hy =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x > 16.0 && y > 70.0 ? z : 0, xs, ys, zs)))
	mean_hx_hy   = mean(hx_hy)
	length_hx_hy = length(hx_hy)
	#-------------------------------------------------------------------
	plot(title="Income2 Data: Split at Inflection Points of Rate k", seriestype=:scatter, legend=:none, xlabel="Education", ylabel="Seniority")
	plot!([(16.0, 0),  (16, 220)])                # vertical line at x_k_max = 16
	plot!([( 9.5, 70), (22,  70)])                # horizontal line at y_k_max = 70
	#-------------------------------------------------------------------------------
	annotate!(20.7,  60, "y_k_max=70", 10)
	annotate!(14.7, 210, "x_k_max=16", 10)
	#-------------------------------------------------------------------------------
	annotate!(11.5,  40, "N=7", 10)
	annotate!(11.5,  25, "mean(z)=24.1", 10)
	annotate!(11.5,  10, "lxly", 10)
	#----------------------------------------
	annotate!(11.5, 140, "N=4", 10)
	annotate!(11.5, 125, "mean(z)=43.8", 10)
	annotate!(11.5, 110, "lxhy", 10)
	#----------------------------------------
	annotate!(19.0,  40, "N=4", 10)
	annotate!(19.0,  25, "mean(z)=69.0", 10)
	annotate!(19.0,  10, "hxly", 10)
	#----------------------------------------
	annotate!(19.0, 140, "N=15", 10)
	annotate!(19.0, 125, "mean(z)=84.1", 10)
	annotate!(19.0, 110, "hxhy", 10)
	#-------------------------------------------------------------------
	# (N_lx_ly=length_lx_ly, N_lx_hy=length_lx_hy, N_hx_ly=length_hx_ly, N_hx_hy=length_hx_hy, mean_lx_ly=mean_lx_ly, mean_lx_hy=mean_lx_hy, mean_hx_ly=mean_hx_ly, mean_hx_hy=mean_hx_hy)
	#-------------------------------------------------------------------------------
end # let

# ╔═╡ f76d77bf-eb40-4f9b-8fca-4b12994d8aef
md"
---
##### 3.5 ISLR2, Fig.2.3 $Income \rightarrow (Education \times Seniority)$
###### 3.5.1 2D-Group-Histogram
"

# ╔═╡ cedc8274-24c4-4fab-bae8-70c5f9276e49
let
	#-------------------------------------------------------------------------------
	xs = income2DataFrame.Education
	ys = income2DataFrame.Seniority
	zs = income2DataFrame.Income
	#-------------------------------------------------------------------------------
	lx_ly =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x <= 16.0 && y <= 70.0 ? (z, :lxly) : 0, xs, ys, zs)))
	#---------------------------------------------------------
	lx_hy =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x <= 16.0 && y > 70.0 ? (z, :lxhy) : 0, xs, ys, zs)))
	#---------------------------------------------------------
	hx_ly =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x > 16.0 && y <= 70.0 ? (z, :hxly) : 0, xs, ys, zs)))
	#---------------------------------------------------------
	hx_hy =
		filter(z -> z !== 0, 
			(map((x, y, z) -> 
				x > 16.0 && y > 70.0 ? (z, :hxhy) : 0, xs, ys, zs)))
	#-------------------------------------------------------------------------------
	gs  = cat(lx_ly, lx_hy, hx_ly, hx_hy; dims=1)
    gzs = [gsi[1] for gsi in gs] 
	ggs = [gsi[2] for gsi in gs]
	groupedhist(gzs, group = ggs, title="Income2 Data: Split at Inflection Points of Rate k", xlabel="Income", bins=10)
	#-------------------------------------------------------------------------------
end # let

# ╔═╡ 467c6c7c-3acc-4919-99d9-4f197a067174
md"
---
###### 3.5.2 Frequency Table and Subgroup Means
"

# ╔═╡ b6dedd81-c882-469d-8bf7-c6abbd8ce724
let
	xs  = income2DataFrame.Education
	ys  = income2DataFrame.Seniority 
	zs  = income2DataFrame.Income
	rXY   = cor(xs, ys)
	CorrelationTest(xs, ys)
end # let

# ╔═╡ 4ea401b0-4604-4ab1-a5c4-cbdf4dc3195e
md"
---
###### 3.5.3 Model-free Data Analysis and Modelling Consequences

From cross-tables and histograms we can see that the rank order in $Income$ is from $lxly$ over $lxhy$ and $hxly$ to $hxhy$. So we see that a low value in $Education$ *cannot* be compensated by $Seniority$ though a higher $Education$ or $Seniority$ *alone* are favorable influences on $Income$ (at least in this *simulated* data set).

So it is a good idea to model the univariate influences by a *nonmonotonic ascending* function. We used linear and nonlinear (sigmoid or logistic) functions. In case of the logistic functions we have the advantage that these are solutions of simple *growth processes* formalized as linear differential equations.

Despite the fact that the frequencies of the two extreme groups $lxly$ and $hxhy$ are higher than in the two other groups the variables $Education$ and $Seniority$ are correlationally *independent*. The corresponding *product-moment correlation* coefficient is $r = 0.19$ (as can be seen above). This is the obligation to construct a nonlinear 3D-regression surface for the multivariate regression model $Income \rightarrow Education \times Seniority$ by multiplying the marginal regression functions of the two marginal nonlinear regression models $Income \rightarrow Education$ and $Income \rightarrow Seniority$.

"

# ╔═╡ 19579a86-97ef-4d18-9a71-89fc350571a8
md"
---
##### 3.6 ISLR2, Fig.2.3 $Income \rightarrow Education \times Seniority$
###### 3.6.1 3D Plot
"

# ╔═╡ 976f54be-4c0f-409b-bf8c-e37d494b472c
let
	#--------------------------------------------------------------------------------
	Plots.plot(title="3D-Plot of 'Income2'-Data (cf. ISLR2, Fig.2.3)", legend=:none, windowsize= (600*2.0, 400*2.0))
	xs  = income2DataFrame.Education
	ys  = income2DataFrame.Seniority 
	zs  = income2DataFrame.Income
	#--------------------------------------------------------------------------------
	rXY   = round(cor(xs, ys), digits=2)
	rXYsQ = round(rXY^2, digits=2)
	#--------------------------------------------------------------------------------
	Plots.plot!(xs, ys, zs, seriestype=:scatter, markersize=8, xlabel="Education", ylabel="Seniority", zlabel="Income", grid=(10))
	#--------------------------------------------------------------------------------
	annotate!(31.8, 20, 110, "r(Edu, Sen)=$rXY", 15)
	annotate!(31.4, 15, 100, "r^2(Edu, Sen)=$rXYsQ", 15)
	annotate!(32.9, 15, 95, "p_corr=0.30", 15)
	#--------------------------------------------------------------------------------
end # let

# ╔═╡ 2ae93ea8-4106-4ff5-b69d-a5d1847e32aa
md"
---
###### 3.6.2 *Multiple* Linear Regression $Income \rightarrow Education \times Seniority$ with 2 Predictors
"

# ╔═╡ 3e40aac3-88b2-4b8c-a003-03e7317d107c
let
	#-------------------------------------------------------------------------------
	xs  = income2DataFrame.Education
	ys  = income2DataFrame.Seniority 
	zs  = income2DataFrame.Income
	#-------------------------------------------------------------------------------
	ols_lin = lm(@formula(Income ~ 1 + Education + Seniority), income2DataFrame)
	yhat = predict(ols_lin)
	#-------------------------------------------------------------------------------
	e    = residuals(ols_lin)
	sse  = round(e'e, digits=2)
	#-------------------------------------------------------------------------------
	rXYhat   = round(cor(xs, yhat), digits=2)
	rXYhatsQ = round(rXYhat^2, digits=2)
	#-------------------------------------------------------------------------------
	Plots.plot(title="3D-Plot of 'Income2'-Data (cf. ISLR2, Fig.2.3)", legend=:none, windowsize= (600*2.0, 400*2.0))
	#-------------------------------------------------------------------------------
	Plots.surface!(xs, ys, yhat, c=:viridis, legend=:none, xlabel="Education", ylabel="Seniority", zlabel="Income", nx=200, ny=200, display_option=Plots.GR.OPTION_SHADED_MESH, grid=(10))
	#-------------------------------------------------------------------------------
	Plots.plot!(xs, ys, zs, seriestype=:scatter, markersize=8, xlabel="Education", ylabel="Seniority", zlabel="Income", grid=(10))
	#-------------------------------------------------------------------------------
	Plots.plot!(map((x, y, z, yh) -> ([x, x], [y, y], [z, yh]), xs, ys, zs, yhat), label="", linewidth = 4)
	#-------------------------------------------------------------------------------
	annotate!(31, 20, 110, "multiple R=$rXYhat", 15)
	annotate!(32.3, 17, 105, "R^2=$rXYhatsQ", 15)
	#-------------------------------------------------------------------------------
end # let

# ╔═╡ 6177f953-55a0-43cb-8d37-4990461b946b
md"
---
###### 3.6.3 *Simple* Linear Regression $Income \rightarrow (Education \cdot Seniority)$ with 1 Predictor

Concerning $R^2$ this is *no* improvement in comparison against the multiple *linear* model (above). But because we have only 2 parameters, we prefer *this* model.
"

# ╔═╡ 98bbe5b4-3811-405f-9965-48885038cac5
let
	#-------------------------------------------------------------------------------
	xs  = income2DataFrame.Education
	ys  = income2DataFrame.Seniority 
	zs  = income2DataFrame.Income
	#-------------------------------------------------------------------------------
	ols_lin = lm(@formula(Income ~ 1 + Education*Seniority), income2DataFrame)
	yhat = predict(ols_lin)
	#-------------------------------------------------------------------------------
	e    = residuals(ols_lin)
	sse  = round(e'e, digits=2)
	#-------------------------------------------------------------------------------
	rXYhat   = round(cor(xs, yhat), digits=2)
	rXYhatsQ = round(rXYhat^2, digits=2)
	#-------------------------------------------------------------------------------
	Plots.plot(title="3D-Plot of 'Income2'-Data (cf. ISLR2, Fig.2.3)", legend=:none, windowsize= (600*2.0, 400*2.0))
	#-------------------------------------------------------------------------------
	Plots.surface!(xs, ys, yhat, c=:viridis, legend=:none, xlabel="Education", ylabel="Seniority", zlabel="Income", nx=200, ny=200, display_option=Plots.GR.OPTION_SHADED_MESH, grid=(10))
	#-------------------------------------------------------------------------------
	Plots.plot!(xs, ys, zs, seriestype=:scatter, markersize=8, xlabel="Education", ylabel="Seniority", zlabel="Income", grid=(10))
	#-------------------------------------------------------------------------------
	Plots.plot!(map((x, y, z, yh) -> ([x, x], [y, y], [z, yh]), xs, ys, zs, yhat), label="", linewidth = 4)
	#-------------------------------------------------------------------------------
	annotate!(31, 20, 110, "multiple R=$rXYhat", 15)
	annotate!(32.3, 17, 105, "R^2=$rXYhatsQ", 15)
	#-------------------------------------------------------------------------------
end # let

# ╔═╡ 60bbd574-5246-4145-819b-f149eedcdcdc
md"
---
###### 3.6.4 NonLinear Surface $Income \rightarrow Education \times Seniority$
"

# ╔═╡ 8082f864-ece5-424d-ac92-93ca8f344ced
let
	xs  = income2DataFrame.Education
	ys  = income2DataFrame.Seniority
	zs  = income2DataFrame.Income
	#-------------------------------------------------------------------------------
	ols_lin = lm(@formula(Income ~ 1 + Education + Seniority), income2DataFrame)
	yhat = predict(ols_lin)
	#-------------------------------------------------------------------------------
	Plots.plot(title="3D-Plot of 'Income2'-Data (cf. ISLR2, Fig.2.3)", legend=:none, windowsize= (600*2.0, 400*2.0))
	#----------------------------------------------------------------
	Plots.surface!(xs, ys, zs, c=:viridis, legend=:none, xlabel="Education", ylabel="Seniority", zlabel="Income", nx=200, ny=200, display_option=Plots.GR.OPTION_SHADED_MESH, grid=(10))
	#-------------------------------------------------------------------------------
	Plots.plot!(xs, ys, zs, seriestype=:scatter, markersize=8, xlabel="Education", ylabel="Seniority", zlabel="Income", grid=(10))
	#-------------------------------------------------------------------------------
	Plots.plot!(map((x, y, z, yh) -> ([x, x], [y, y], [z, yh]), xs, ys, zs, yhat), label="", linewidth = 4)
	#-------------------------------------------------------------------------------
end # let

# ╔═╡ 9bc9fe72-7dee-495e-9e0a-cc86ae653092
md"
---

###### 3.6.5 Nonlinear Bivariate Model using the [Generalized Logistic Function](https://en.wikipedia.org/wiki/Generalised_logistic_function)

The *marginal* (*univariate*) model functions are:

$f(X) = L_X + \frac{U_X - L_X}{1+ e^{-k_X \left(X - x_{k_{X_{max}}} \right)}},$

and

$f(Y) = L_Y + \frac{U_Y-L_Y}{1+e^{-k_Y\left(Y - y_{k_{Y_{max}}}\right)}}.$

where:
- **$k_{x_{max}}$** = $max(k_x)$ = max. growth rate $k_x$ of $X$
- **$k_{y_{max}}$** = $max(k_y)$ = max. growth rate $k_y$ of $Y$
- **$U$** = Upper (right) asymptote of $X$ and $Y$
- **$L$** = Lower (left) asymptote of $X$ and $Y$
- **$x_{k_{max}},y_{k_{max}}$** = $\underset{X,Y}{\operatorname{argmax}\;k} \;\; \text{(= }x \text{ and } y\text{ are the inflection points of rates } k$ of $X$ and $Y)$  

The *bivariate* model function is: 

$f(X, Y) = \frac{f(X)f(Y)}{c}.$

"

# ╔═╡ e354fade-e1d8-4bd2-8e47-ec4046354c55
md"

where $c$ is a normalizing factor so that the predicted range approximates the range of the dependent variable $Z$.

The product of the *marginal* functions is possible, because the correlation $r_{X,Y}$ is not significant. So the predictor variables $Education$ and $Seniority$ are *independent*.
"

# ╔═╡ 9c286327-ffad-41bf-85bf-e843a3cedcf0
let    
	#--------------------------------------------------------------------------------
	zs  = income2DataFrame.Income
	xs  = [x for x in range(9, 22; length=30)]
	ys  = [y for y in range(9, 200; length=30)]
	#--------------------------------------------------------------------------------
	# parmsX[1] = L (= Lower (left) asymptote)
	# prmsX[2] = U (= Upper (right) asymptote)
	# prmsX[3] = k (= growth rate)
	# prmsX[4] = x_k_max (= x of max k = infection point of k
	#--------------------------------------------------------------------------------
	# myModel(xs, p) = p[1] .+ (p[2] .- p[1])./((1 .+ exp.(-p[3] .* (xs .- p[4]))))
	myModel(x, p) = p[1] + (p[2] - p[1])/((1 + exp(-p[3] * (x - p[4]))))
	#--------------------------------------------------------------------------------
	function myModelXY(x, y) 
		parmsX = [28.0, 88.0, 0.9, 16.0]          # initial estimates of parms
		zhatX = myModel(x, parmsX)
		#----------------------------------------------------------------------
		parmsY = [40.0, 78.0, 4.0, 70.0]          # initial estimates of parms
		zhatY = myModel(y, parmsY)
		#----------------------------------------------------------------------
		zhatXY = (zhatX * zhatY)/17.2
		#----------------------------------------------------------------------
	end # function myModelXY
	#-------------------------------------------------------------------------------
	fs = [myModelXY(x, y) for x in xs for y in ys]
	#-------------------------------------------------------------------------------
	Plots.plot(title="3D-Wireframe of ISLR2-Data 'Income2' (cf.Fig.2.3)", legend=:none, windowsize= (600*2.0, 400*2.0))
	#-------------------------------------------------------------------------------
	Plots.wireframe!(xs, ys, fs, legend=:none, xlabel="Education", ylabel="Seniority", zlabel="Income", nx=200, ny=200, camera=(40, 20))
	#---------------------------------------------------------------
	# Plots.wireframe!(xs, ys, fs, legend=:none, xlabel="Education", ylabel="Seniority", zlabel="Income", nx=200, ny=200, color=:viridis, display_option=Plots.GR.OPTION_SHADED_MESH,  camera=(40, 20))
	#-------------------------------------------------------------------------------
	annotate!(31, 20, 310, "r(Edu, Sen)=0.19", 15)
	annotate!(32.5, 15, 295, "p_corr=0.30", 15)
	#-------------------------------------------------------------------------------
end # let

# ╔═╡ f768add3-38aa-40e1-bc95-661f380a817d
let    
	#--------------------------------------------------------------------------------
	# xs  = income2DataFrame.Education
	# ys  = income2DataFrame.Seniority
	zs  = income2DataFrame.Income
	xs  = [x for x in range(9, 22; length=30)]
	ys  = [y for y in range(9, 200; length=30)]
	#--------------------------------------------------------------------------------
	# parmsX[1] = L (= Lower (left) asymptote)
	# prmsX[2] = U (= Upper (right) asymptote)
	# prmsX[3] = k (= growth rate)
	# prmsX[4] = x_k_max (= x of max k = infection point of k
	#--------------------------------------------------------------------------------
	# myModel(xs, p) = p[1] .+ (p[2] .- p[1])./((1 .+ exp.(-p[3] .* (xs .- p[4]))))
	myModel(x, p) = p[1] + (p[2] - p[1])/((1 + exp(-p[3] * (x - p[4]))))
	#--------------------------------------------------------------------------------
	function myModelXY(x, y) 
		parmsX = [28.0, 88.0, 0.9, 16.0]          # initial estimates of parms
		zhatX = myModel(x, parmsX)
		#----------------------------------------------------------------------
		parmsY = [40.0, 78.0, 4.0, 70.0]          # initial estimates of parms
		zhatY = myModel(y, parmsY)
		#----------------------------------------------------------------------
	    # zhatXY = (zhatX * zhatY) / 100.0
		# zhatXY = (zhatX/22.0 * zhatY/200) * 100.0
		zhatXY = (zhatX * zhatY)/17.2
		#----------------------------------------------------------------------
	end # function myModelXY
	#--------------------------------------------------------------------------------
	fs = [myModelXY(x, y) for x in xs for y in ys]
	#--------------------------------------------------------------------------------
	Plots.plot(title="3D-Wireframe of ISLR2-Data 'Income2' (cf.Fig.2.3)", legend=:none, windowsize= (600*2.0, 400*2.0))
	#---------------------------------------------------------------
	# Plots.wireframe!(xs, ys, fs, legend=:none, xlabel="Education", ylabel="Seniority", zlabel="Income", nx=200, ny=200, camera=(40, 20))
	#---------------------------------------------------------------
	Plots.wireframe!(xs, ys, fs, legend=:none, xlabel="Education", ylabel="Seniority", zlabel="Income", nx=200, ny=200, color=:viridis, display_option=Plots.GR.OPTION_SHADED_MESH,  camera=(40, 20))
    #--------------------------------------------------------------------------
	annotate!(31, 20, 310, "r(Edu, Sen)=0.19", 15)
	annotate!(32.5, 15, 295, "p_corr=0.30", 15)
	#--------------------------------------------------------------------------
end # let

# ╔═╡ 2ad1fc50-3505-4a58-a248-c9453df6cc01
md"
---
##### 3.7 Appendix: Derivatives of Sigmoid (= Logistic) Functions

###### 3.7.1 Derivatives of the simple [*Sigmoid* (= *Logistic*) function](https://en.wikipedia.org/wiki/Logistic_function)
"

# ╔═╡ c3e20f84-a4d4-4a93-8885-191ca44baf1a
md"

The simple *sigmoid* (= *logistic*) model function $(0 < f(x) < 1)$ is:

$f(x) = \frac{1}{1+e^{-x}}=\frac{e^x}{e^x(1+e^{-x})}=\frac{e^x}{e^x+e^0}=\frac{e^x}{e^x+1},$ 


and


$1-f(x) = \frac{e^x+1}{e^x+1} - \frac{e^x}{e^x+1} = \frac{1}{e^x+1}.$

$\;$

$\;$

Applying the *quotient rule* the derivation is either

$f_1'(x) = \frac{0+e^{-x}}{(1+e^{-x})^2}=\frac{e^{-x}}{(1+e^{-x})^2}$

or

$f_2'(x) = \frac{d}{dx}f(x) = \frac{e^x(e^x+1)-e^xe^x}{(e^x+1)^2} = \frac{e^{2x}+e^x-e^{2x}}{(e^x+1)^2} = \frac{e^x}{(e^x+1)^2}.$

$\;$

$\;$

$\;$


The *second* case $f_2'(x)$ can be simplified to the product of two terms:



$f_2'(x) = \frac{e^x}{(e^x+1)^2} = \frac{e^x}{(e^x+1)(e^x+1)} = \frac{e^x}{(e^x+1)} \cdot \frac{1}{(e^x+1)} = f(x)(1-f(x)).$

$\;$

$\;$

$\;$

Another alternative is known as (WolframAlpha, 2023):

$f_3'(x) = \frac{1}{(e^x+1)} - \frac{1}{(e^x+1)^2}.$

$\;$

$\;$

The decomposition $f_2'(x) = f(x)(1-f(x))$ is known as the [*logistic differential equation*](https://en.wikipedia.org/wiki/Logistic_function) with boundary conditions

$\;$

$\;$

$f(0)= \frac{1}{2} \text{ and } f'(0) = \frac{1}{4}.$

$\;$

$\;$

The differential equation is used for the description of simple *growth* or *learning* processes with saturation.

"


# ╔═╡ 74b5d35b-cbf2-4e89-8cc6-19ade90c6e04
let
	x_k_max = 0
	L = 0
	plot(title="Simple Logistic Function f(x) = 1/(1 + exp(-x))")
	f(x) = exp(x-x_k_max)/(1+exp(x-x_k_max))
	plot!([x for x in -10:0.1:10], [f(x) for x in -10:0.1:10])
	plot!([x_k_max, x_k_max],[L, f(x_k_max)], legend=:none)        # vertical line
	plot!([-10, 0],[f(x_k_max), f(x_k_max)], legend=:none)         # horizntal line
end # let

# ╔═╡ 2ccd9256-e5e2-4b18-8468-11ab991e01bd
let
	plot(title="Derivative fDeriv1(x) of the Simple Logistic Function")
	fDeriv1(x) = exp(-x)/(1+exp(-x)^2)
	fDeriv2(x) = exp(x)/(1+exp(x)^2)
	fDeriv3(x) = 1/(exp(x)+1) - 1/(exp(x)+1)^2
	plot!([x for x in -10:0.1:10], [fDeriv1(x) for x in -10:0.1:10])
end # let

# ╔═╡ 6624fd5f-3d74-4607-a090-a61535712eec
let
	plot(title="Derivative fDeriv2(x) of the Simple Logistic Function")
	fDeriv1(x) = exp(-x)/(1+exp(-x)^2)
	fDeriv2(x) = exp(x)/(1+exp(x)^2)
	fDeriv3(x) = 1/(exp(x)+1) - 1/(exp(x)+1)^2
	plot!([x for x in -10:0.1:10], [fDeriv2(x) for x in -10:0.1:10])
end # let

# ╔═╡ e015f72c-c716-4ca5-8b84-4ef4f43575ab
let
	plot(title="Derivative fDeriv3(x) of the Simple Logistic Function")
	fDeriv1(x) = exp(-x)/(1+exp(-x)^2)
	fDeriv2(x) = exp(x)/(1+exp(x)^2)
	fDeriv3(x) = 1/(exp(x)+1) - 1/(exp(x)+1)^2
	plot!([x for x in -10:0.1:10], [fDeriv3(x) for x in -10:0.1:10])
end # let

# ╔═╡ 9a9676be-cc57-4814-bfff-08b6cf0ed9f7
md"
---
###### 3.7.2 Derivative of the [*Generalized Sigmoid* (= *Logistic*) function](https://en.wikipedia.org/wiki/Generalised_logistic_function)

"

# ╔═╡ 08b35085-0232-4fc9-83d9-2e444f5f587f
###### 3.7.2 Derivative of our [Generalized Logistic Function](https://en.wikipedia.org/wiki/Generalised_logistic_function) 

md"
---

$f(x) = L + (U-L)\cdot\frac{1}{1+e^{-k\left(x - x_{k_{max}}\right)}}$

where:
- **$k_{max}$** = $max(k)$ = max. growth rate *k* of $X$
- **$U$** = Upper (right) asymptote of $X$
- **$L$** = Lower (left) asymptote of $X$
- **$x_{k_{max}}$** = $\underset{X}{\operatorname{argmax}\;k} \;\; \text{(= }x\text{ is the inflection point of } k)$ 

$\;$

The original output of $expand\_derivatives(Dx(myModel))$ is:

$f'(x)=\frac{d}{dx}f(x)=0.58\left(\frac{59.36}{(1+e^{-0.58(-15.6+x)})^2} \right)\cdot e^{-0.58(-15.6+x)}.$

$\;$

$\;$

$\;$

When we rearrange terms the similarity with $f_2'(x)$ (above) becomes obvious.

$\;$

$f'(x)=\frac{d}{dx}f(x)=(0.58\cdot 59.36)\left(\frac{e^{-0.58(-15.6+x)}}{(1+e^{-0.58(-15.6+x)})^2} \right)$

$\;$

$\;$

$\;$

$=34.4288\left(\frac{e^{-0.58(-15.6+x)}}{(1+e^{-0.58(-15.6+x)})^2} \right)$

$\;$

$\;$

$\;$

with *boundary conditions*:

$\;$

$f(x_{k_{max}}) = 18.89 + (78.25-18.89)\cdot\frac{1}{2} = 48.57 \text{ and } f'(x_{k_{max}}) = 34.4288\cdot\frac{1}{4} = 8.6072$
" 

# ╔═╡ 951da7e0-e296-4ef5-b568-ad81ecf790e9
md"
The abstracted form of $f'(x)$ is:

$f'(x) = k \cdot (U-L)\cdot \left(\frac{e^{-k\left(x-x_{k_{max}}\right)}}{\left(1+e^{-k\left(x-x_{k_{max}}\right)}\right)^2} \right)$

$\;$

$\;$

with *boundary conditions*

$\;$

$\;$

$f(x_{k_{max}}) = L + (U-L)\cdot\frac{1}{2} \text{ and } f'(x_{k_{max}}) = k\cdot (U-L)\cdot\frac{1}{4}$

"

# ╔═╡ abf63c1c-d799-4aac-9dde-fc22cc963a9b
let
	x_k_max = 15.65
	k = 0.58
	U = 78.25
	L = 18.89
	plot(title="Gen.Log.Fun. f(x) = L+(U-L)/(1 + exp(-k(x-x_k_max))")
	annotate!(5, 76, "Income1 Data Set")
	f(x) = L + (U-L)/(1 + exp(-k*(x - x_k_max)))
	plot!([x for x in 0:0.1:30], [f(x) for x in 0:0.1:30], legend=:none)
	plot!([x_k_max, x_k_max],[L, f(x_k_max)], legend=:none)        # vertical line
	plot!([0, x_k_max],[f(x_k_max), f(x_k_max)], legend=:none)     # horizntal line
end # let

# ╔═╡ 5de8dbf0-95ca-4ad9-8c16-a156721c7ae9
let
	#--------------------------------------------------------------------------------
	plot(title="Derivative of Gener. Sigmoid (= Logistic Diff. Equation)")
	@variables x
	L = 18.89        # Lower left asymptote
	U = 78.25        # Upper right asymptote
	k =  0.58        # max rate k
	x_k_max = 15.65  # arg_max k = that x for max k = x|max(k)
	#--------------------------------------------------------------------------------
	myModel = L + (U - L)*(1/((1 + exp(-k*(x - x_k_max)))))
	Dx = Differential(x)
	dmyModelToX = Dx(myModel)
	dglx = expand_derivatives(Dx(myModel))
	expExpression(x) = exp(-k*(-15.65+x))
	dgl_myModel(x) = 0.58*(59.36/(1+expExpression(x))^2)*expExpression(x)
	#--------------------------------------------------------------------------------
	plot!([x for x in 0.65:0.1:15.65*2], [dgl_myModel(x) for x in 0.65:0.1:15.65*2])
	#--------------------------------------------------------------------------------
end # let

# ╔═╡ e13cda08-84ec-48c3-ade8-fd61a3c5ac17
md"
---
##### References
- **James, G., Witten, D., Hastie, T. & Tibshirani, R.**; *An Introduction to Statistical Learning, Heidelberg: Springer, 2021, 2/e, 
- **Wikipedia**; *Logistic Function*; [https://en.wikipedia.org/wiki/Logistic_function](https://en.wikipedia.org/wiki/Logistic_function); last visit 2023/05/07
- **Wikipedia**; *The Generalized Logistic Function*, [https://en.wikipedia.org/wiki/Generalised_logistic_function](https://en.wikipedia.org/wiki/Generalised_logistic_function); last visit 2023/05/04
- **WolframAlpha**; Version 1.4.19.2022041167; last visit 23/05/04
"

# ╔═╡ 75c6cefe-7be7-4086-a5c6-0e8312b8b6ee
md"
---
##### end of ch. 2.1
"

# ╔═╡ 6566d57d-d090-4bed-87e8-be8b516e86e1
md"
====================================================================================

This is a **draft** under the Attribution-NonCommercial-ShareAlike 4.0 International **(CC BY-NC-SA 4.0)** license. Comments, suggestions for improvement and bug reports are welcome: **claus.moebus(@)uol.de**

====================================================================================
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
HypothesisTests = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
LsqFit = "2fda8390-95c7-5789-9bda-21331edee243"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
CSV = "~0.10.9"
GLM = "~1.8.3"
HypothesisTests = "~0.10.12"
LsqFit = "~0.13.0"
Plots = "~1.38.11"
StatsPlots = "~0.15.5"
Symbolics = "~5.3.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "7802db3b7400f501375e0842b626ba779969533a"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "3ee5c58774f4487a5bf2bb05e39d91ff5022b4cc"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.29.4"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "38911c7737e123b28182d89027f4216cfc8a9da7"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bijections]]
git-tree-sha1 = "fe4f8c5ee7f76f2198d5c2a06d3961c249cce7bd"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.4"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "SnoopPrecompile", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "c700cce799b51c9045473de751e9319bdd1c6e94"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "f84967c4497e0e1955f9a582c232b02847c5f589"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.7"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "a6e6ce44a1e0a781772fc795fb7343b1925e9898"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "9441451ee712d1aec22edad62db1a9af3dc8d852"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "b306df2650947e9eb100ec125ff8c65ca2053d30"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c2614fa3aafe03d1a44b8e16508d9be718b8095a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.89"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "698124109da77b6914f64edd696be8dccf90229e"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "8b84876e31fa39479050e2d3395c4b3b210db8b0"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.4.6"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f9818144ce7c8c41edf5c4c179c684d92aa4d9fe"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.6.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "fc86b4fd3eff76c3ce4f5e96e2fdfa6282722885"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.0.0"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "6604e18a0220650dbbea7854938768f15955dd8e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.20.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "97829cfda0df99ddaeaafb5b370d6cab87b7013e"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "efaac003187ccc71ace6c755b197284cd4811bfe"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4486ff47de4c18cb511a0da420efebb314556316"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.4+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "Logging", "MultivariatePolynomials", "Primes", "Random", "SnoopPrecompile"]
git-tree-sha1 = "b6c3e9e1eb8dcc6fd9bc68fe08dcc7ab22710de6"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.3.4"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9e1a5e9f3b81ad6a5c613d181664a0efc6fe6dd7"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "69182f9a2d6add3736b7a06ab6416aafdeec2196"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.8.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "432b5b03176f8182bd6841fbfc42c718506a2d5f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.15"

[[deps.HypothesisTests]]
deps = ["Combinatorics", "Distributions", "LinearAlgebra", "Random", "Rmath", "Roots", "Statistics", "StatsBase"]
git-tree-sha1 = "3eaee0f574ae7918e0529ed37a2652c6c17d4948"
uuid = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
version = "0.10.12"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "f366daebdfb079fd1fe4e3d560f99a0c892e15bc"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "LinearAlgebra", "MacroTools", "PreallocationTools", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "cd04158424635efd05ff38d5f55843397b7416a9"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.14.0"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "099e356f267354f46ba65087981a77da23a279b7"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LsqFit]]
deps = ["Distributions", "ForwardDiff", "LinearAlgebra", "NLSolversBase", "OptimBase", "Random", "StatsBase"]
git-tree-sha1 = "00f475f85c50584b12268675072663dfed5594b2"
uuid = "2fda8390-95c7-5789-9bda-21331edee243"
version = "0.13.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "eaa98afe2033ffc0629f9d0d83961d66a021dfcc"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.7"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "964cb1a7069723727025ae295408747a0b36a854"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.3.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "7fb975217aea8f1bb360cf1dde70bad2530622d2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OptimBase]]
deps = ["NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "9cb1fee807b599b5f803809e85c81b582d2009d6"
uuid = "87e2bd06-a317-5318-96d9-3ecbac512eee"
version = "2.0.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6c7f47fd112001fc95ea1569c2757dffd9e81328"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.11"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "Requires"]
git-tree-sha1 = "f739b1b3cc7b9949af3b35089931f2b58c289163"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.12"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "d0984cc886c48e5a165705ce65236dc2ec467b91"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "311a2aa90a64076ea0fac2ad7492e914e6feeb81"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "062986376ce6d394b23d5d90f01d81426113a3c9"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.3"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "6d7bb727e76147ba18eed998700998e17b8e4911"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.4"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "68078e9fa9130a6a768815c48002d0921a232c11"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.38.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "e19c09f5cc868785766f86435ba40576cf751257"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.14"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "f139e81a81e6c29c40f1971c9e5309b09c03f2c3"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.6"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SnoopPrecompile", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
git-tree-sha1 = "392d3e28b05984496af37100ded94dc46fa6c8de"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.91.7"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "e61e48ef909375203092a6e83508c8416df55a83"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.2.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "c262c8e978048c2b095be1672c9bee55b4619521"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.24"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8cc7a5385ecaa420f0b3426f9b0135d0df0638ed"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.2"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "14ef622cf28b05e38f8af1de57bc9142b03fbfe3"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.5"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SymbolicIndexingInterface]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "f8ab052bfcbdb9b48fad2c80c873aa0d0344dfe5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.2.2"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TimerOutputs", "Unityper"]
git-tree-sha1 = "5cb1f963f82e7b81305102dd69472fcd3e0e1483"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "1.0.5"

[[deps.Symbolics]]
deps = ["ArrayInterface", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "Groebner", "IfElse", "LaTeXStrings", "LambertW", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Markdown", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TreeViews"]
git-tree-sha1 = "e23ec62c083ca8f15a4b7174331b3b8d1c511e47"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "5.3.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "7bc1632a4eafbe9bd94cf1a784a9a4eb5e040a91"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "d5f4ec8c22db63bd3ccb239f640e895cfde145aa"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.2"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─faef2569-8983-4704-9e1a-7080fb16dd06
# ╠═849dbc25-8dd1-416b-b706-672e826eec6a
# ╟─622d27ff-e36f-4b69-949d-1fa78a8413ce
# ╟─7b047630-b637-47ee-9aee-6d685dd3e773
# ╟─f698b886-e349-436f-932d-657bf7ff1eb5
# ╟─de0c31cf-a954-41fb-afca-a03cf28364dc
# ╟─b852829d-0da2-44db-b7ed-9bf390858882
# ╟─009a4323-0593-4900-b0d9-e2c4fc7a8756
# ╟─4d9fc2b5-3ee4-4acf-be66-d2881f9f994a
# ╟─da5a6876-ef9f-4d73-bd3b-1369c70bc103
# ╟─24576004-b1d2-43e1-a861-eba45d1fda0e
# ╟─1feee341-401b-4afd-8970-7c2804fc3f89
# ╟─8f6de19f-c4ff-46dd-a120-269ebf647578
# ╟─d800b01d-477b-4776-bd2c-848fff8a5da9
# ╟─7ad28ec0-74ad-4f15-8235-259f091622a6
# ╟─55a0b4d7-890f-4d32-a10f-002631906bda
# ╟─bfcdfaf5-50c1-452a-946b-941e5c4c1041
# ╟─5479bd68-aafa-47c6-bce5-d5a8817c131a
# ╟─92ea0bb5-d318-4758-83c9-ac5960918e12
# ╟─e5a0b52d-f482-43fa-9efc-84ecd0711794
# ╟─5adf8b34-2d41-4b15-8c13-0cbf84ed0cce
# ╟─ec30ce62-3699-40f8-88c3-8e394d6d826e
# ╟─c54e5513-f30a-4bc1-afe0-102ba43a6f87
# ╟─a559d47a-bdf9-4229-b363-d33ff4994de6
# ╟─a0e54499-eb79-4381-81eb-acfd8826511e
# ╟─314821cf-1739-454e-a6eb-75b3bed752b3
# ╟─d1982304-7008-4486-a16d-5d4064935ebc
# ╟─09af8231-726f-4439-b123-a14170b5170e
# ╟─df58736c-ebd1-4020-915c-b32291985f3f
# ╟─8e78b3e0-e97e-4b8d-9342-a180b26bbf03
# ╟─2771a9f1-353f-4bfa-b4d9-a0deda6e031a
# ╟─84c06bd1-4dbf-411f-96c4-97554146ca21
# ╟─baa67da7-d394-4a1d-9eee-3a982c0f74c2
# ╟─f3025a7a-0256-46c7-b296-4cc3a36c5372
# ╟─e31097bf-1cbf-4146-91b3-841b91aa11b3
# ╟─f98e88e4-6abf-41c6-86c4-60f61523e1bb
# ╟─1e5f39c1-b6e0-4088-b0dd-5e6162602bed
# ╟─4da9634c-41ab-469f-a25b-dc618e65eacb
# ╟─f76d77bf-eb40-4f9b-8fca-4b12994d8aef
# ╟─cedc8274-24c4-4fab-bae8-70c5f9276e49
# ╟─467c6c7c-3acc-4919-99d9-4f197a067174
# ╟─b6dedd81-c882-469d-8bf7-c6abbd8ce724
# ╟─4ea401b0-4604-4ab1-a5c4-cbdf4dc3195e
# ╟─19579a86-97ef-4d18-9a71-89fc350571a8
# ╟─976f54be-4c0f-409b-bf8c-e37d494b472c
# ╟─2ae93ea8-4106-4ff5-b69d-a5d1847e32aa
# ╟─3e40aac3-88b2-4b8c-a003-03e7317d107c
# ╟─6177f953-55a0-43cb-8d37-4990461b946b
# ╟─98bbe5b4-3811-405f-9965-48885038cac5
# ╟─60bbd574-5246-4145-819b-f149eedcdcdc
# ╟─8082f864-ece5-424d-ac92-93ca8f344ced
# ╟─9bc9fe72-7dee-495e-9e0a-cc86ae653092
# ╟─e354fade-e1d8-4bd2-8e47-ec4046354c55
# ╟─9c286327-ffad-41bf-85bf-e843a3cedcf0
# ╟─f768add3-38aa-40e1-bc95-661f380a817d
# ╟─2ad1fc50-3505-4a58-a248-c9453df6cc01
# ╟─c3e20f84-a4d4-4a93-8885-191ca44baf1a
# ╟─74b5d35b-cbf2-4e89-8cc6-19ade90c6e04
# ╟─2ccd9256-e5e2-4b18-8468-11ab991e01bd
# ╟─6624fd5f-3d74-4607-a090-a61535712eec
# ╟─e015f72c-c716-4ca5-8b84-4ef4f43575ab
# ╟─9a9676be-cc57-4814-bfff-08b6cf0ed9f7
# ╟─08b35085-0232-4fc9-83d9-2e444f5f587f
# ╟─951da7e0-e296-4ef5-b568-ad81ecf790e9
# ╟─abf63c1c-d799-4aac-9dde-fc22cc963a9b
# ╟─5de8dbf0-95ca-4ad9-8c16-a156721c7ae9
# ╟─e13cda08-84ec-48c3-ade8-fd61a3c5ac17
# ╟─75c6cefe-7be7-4086-a5c6-0e8312b8b6ee
# ╟─6566d57d-d090-4bed-87e8-be8b516e86e1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
