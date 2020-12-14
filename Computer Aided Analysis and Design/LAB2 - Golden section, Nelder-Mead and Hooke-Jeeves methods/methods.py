import math, copy

def f(x):
	return x**2 - 7*x + 12

def f1(x):
    y = 100 * math.pow((x[1] - math.pow(x[0],2)),2) + math.pow((1 - x[0]),2)
    return float(y)

def f2(x):
    y = math.pow((x[0] - 4), 2) + 4 * math.pow((x[1] - 2), 2)
    return y

def f3(x):
    y = 0
    for i in range(len(x)):
        y += math.pow((x[i] - (i+1)),2)
    return float(y)

def f4(x):
    y = abs((x[0]-x[1]) * (x[0]+x[1])) + math.pow( (math.pow(x[0],2) + math.pow(x[1],2)), 0.5)
    return float(y)

def f5(x):
    sum = 0
    for i in x:
        sum += i**2
    y = 0.5 + (pow(math.sin(math.sqrt(sum)), 2) - 0.5) / pow(1 + 0.001 * sum, 2)
    return float(y)


def unimodalni(f, tocka, h=1):
	m = copy.deepcopy(tocka)

	l = tocka - h
	r = tocka + h
	step = 1

	fm = f(tocka)
	fl = f(l)
	fr = f(r)

	if (fm < fr and fm < fl):
		return [l, r]
	
	if fm > fr:
		while(fm > fr):
			l = m
			m = r
			fm = fr
			step *= 2
			r = tocka + h*step
			fr = f(r)
		return [l,r]

	else:
		while(fm > fl):
			r = m
			m = l
			fm = fl
			step *= 2
			l = tocka - h*step
			fl = f(l)
		return [l,r]


def zlatni_rez(f, interval=None, tocka=None, e=10**-6):
	if tocka:
		interval = unimodalni(f, tocka)

	k = 0.5*(math.sqrt(5) - 1)

	a = interval[0]
	b = interval[1]
	c = b - k * (b - a)
	d = a + k * (b - a)

	fc = f(c)
	fd = f(d)

	while b - a > e:
		if(fc < fd):
			b = d
			d = c
			c = b - k * (b - a)
			fd = fc
			fc = f(c)

		else:
			a = c
			c = d
			d = a + k * (b - a)
			fc = fd
			fd = f(d)

	return (a + b)/2


def modul_vektora(x):
	suma = 0
	for i in range(len(x)):
		suma += pow(x[i],2)
	return math.sqrt(suma)


def oduzmi_vektore(x1,x2):
	x = len(x1) * [0]
	for i in range(len(x)):
		x[i] = x1[i] - x2[i]
	return x


def koordinatno_trazenje(f, X0, eps=10**-6):
	X = copy.deepcopy(X0)
	e = [[1 if j==i else 0 for j in range(len(X))] for i in range(len(X))]

	while True:
		Xs = copy.deepcopy(X)
		for i in range(len(X)):

			lambda_min = zlatni_rez(lambda l : f([X[j] + l*e[i][j] for j in range(len(X))]), tocka=1)

			X[i] += lambda_min

		if(modul_vektora(oduzmi_vektore(X,Xs)) <= eps):
			break

	return X


def istrazi(f, x, dx):
    F = f(x)
    xs = copy.deepcopy(x)
    for i in range(len(x)):
        xs[i] += dx
        N = f(xs)
        if(N > F):
            xs[i] -= 2*dx
            N = f(xs)
            if(N > F):
                xs[i]+= dx
    return xs


def Hook_Jeeves(f, x0, dx=1, e = 10**-6):
    xp = copy.deepcopy(x0)
    xb = copy.deepcopy(x0)

    while(dx > e):
        xn = istrazi(f, xp, dx)
        if(f(xn) < f(xb)):
            for i in range(len(xp)):
                xp[i] = 2*xn[i] - xb[i]
            xb = copy.deepcopy(xn)
        else:
            dx /= 2.
            xp = copy.deepcopy(xb)
    return xb


def simplex(f, x0, k=1, alfa=1, beta=0.5, gamma=2, eps=10**-6):
	X = []
	for i in range(len(x0)):
		X.append([x0[j]+k if j==i else x0[j] for j in range(len(x0))])
	X.append(x0)

	while True:

		values = [f(X[i]) for i in range(len(X))]
		h = values.index(max(values))
		l = values.index(min(values))

		tmp = copy.deepcopy(X)			# CENTROID
		Xh = tmp.pop(h)
		Xc = [sum(x) for x in zip(*tmp)]
		Xc = [x/len(tmp) for x in Xc]

		Xr = refleksija(Xc, Xh, alfa)
		
		if f(Xr) < f(X[l]):
			Xe = ekspanzija(Xc, Xr, gamma)
			if f(Xe) < f(X[l]):
				X[h] = Xe
			else:
				X[h] = Xr
		else:
			state = True
			for j in range(len(X)):
				if j == h:
					continue
				if f(Xr) < f(X[j]):
					state = False
			if state:
				if f(Xr) < f(X[h]):
					X[h] = Xr

				Xk = kontrakcija(Xc, Xh, beta) 

				if f(Xk) < f(X[h]):
					X[h] = Xk

				else:
					for i in range(len(X)):
						X[i] = [0.5*(X[i][j] + X[l][j]) for j in range(len(X[0]))]
			else:
				X[h] = Xr

		if stop(f, X, Xc, eps):
			break

	return Xc


def refleksija(Xc, Xh, alfa):
	return [(1+alfa)*Xc[i] - alfa*Xh[i] for i in range(len(Xc))]

def ekspanzija(Xc, Xr, gamma):
	return [(1-gamma)*Xc[i] + gamma*Xr[i] for i in range(len(Xc))]

def kontrakcija(Xc, Xh, beta):
	return [(1-beta)*Xc[i] + beta*Xh[i] for i in range(len(Xc))]

def stop(f, X, Xc, eps):
	values = 0
	for i in range(len(X)):
		values += math.pow(f(X[i]) - (f(Xc)),2)
	values = math.sqrt(values/len(X)) 
	return values <= eps