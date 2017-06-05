
from math import exp
from random import uniform
from math import pow

class Backpropagation():

    yv = {'h1': 0, 'h2': 0, 'o1': 0}
    w = {'x0h1':uniform(-1,1),'x0h2':uniform(-1,1),'x0o1':uniform(-1,1),'x1h1':uniform(-1,1),'x1h2':uniform(-1,1),'x2h1':uniform(-1,1),'x2h2':uniform(-1,1),'h1o1':uniform(-1,1),'h2o1':uniform(-1,1)}
    # w = {'x0h1': 0.9301116024382323, 'x0h2': 0.967244646977224, 'x0o1': 0.1254426425449333, 'x1h1': 0.7871505465009867,
    #  'x1h2': 0.026718658290475483, 'x2h1': -0.9028462207827304, 'x2h2': 0.3246247997227605, 'h1o1': -0.3128303186613406,
    #  'h2o1': -0.09324858383870716}
    u = {'h1':0,'h2':0,'o1':0}
    d = {'h1':0,'h2':0,'01':0}
    D = {'x0h1':0,'x0h2':0,'x0o1':0,'h1o1':0,'h2o1':0,'x1h1':0,'x1h2':0,'x2h1':0,'x2h2':0}
    G = {'x0h1':0,'x0h2':0,'x0o1':0,'h1o1':0,'h2o1':0,'x1h1':0,'x1h2':0,'x2h1':0,'x2h2':0}

    def f(self, u):
        return 1/(1+exp(1) ** -u)

    def y(self,padrao):
        self.u['h1'] = self.w['x0h1'] + padrao[0] * self.w['x1h1'] + padrao[1]*self.w['x2h1']
        self.yv['h1'] = self.f(self.u['h1'])

        self.u['h2'] = self.w['x0h2'] + padrao[0] * self.w['x1h2'] + padrao[1] * self.w['x2h2']
        self.yv['h2'] = self.f(self.u['h2'])

        self.u['o1'] = self.w['x0o1'] + self.yv['h1'] * self.w['h1o1'] + self.yv['h2'] * self.w['h2o1']
        self.yv['o1'] = self.f(self.u['o1'])

        return self.yv['o1']

    def mse(self,padroes,desejado):
        soma = 0

        for i in range(0,4):
            soma += ((desejado[i] - self.y(padroes[i])) ** 2)

        return soma/4

    """
        eta = taxa de aprendizado
    """
    def treinar(self,padroes,desejado,eta,alpha):
        epoca = 0

        while True:
            print('Pesos:',self.w)
            print('Erro:',self.mse(padroes, desejado))
            if(self.mse(padroes,desejado) < 0.01 or epoca > 10000 ):
                break

            epoca+=1

            for i in range(0,4):
                y = self.y(padroes[i])
                E = desejado[i] - y
                #deltas
                self.d['o1'] = y * (1-y)*E
                self.d['h1'] = (self.yv['h1'] * (1-self.yv['h1'])) * self.w['h1o1'] * self.d['o1']
                self.d['h2'] = (self.yv['h2'] * (1-self.yv['h2'])) * self.w['h2o1'] * self.d['o1']

                #gradientes
                self.G['x0o1'] = self.d['o1']
                self.G['h1o1'] = self.d['o1'] * self.yv['h1']
                self.G['h2o1'] = self.d['o1'] * self.yv['h2']

                self.G['x0h1'] = self.d['h1']
                self.G['x1h1'] = self.d['h1'] * padroes[i][0]
                self.G['x2h1'] = self.d['h1'] * padroes[i][1]

                self.G['x0h2'] = self.d['h2']
                self.G['x1h2'] = self.d['h2'] * padroes[i][0]
                self.G['x2h2'] = self.d['h2'] * padroes[i][1]

                #atualização
                self.D['x0o1'] = eta*self.G['x0o1']+alpha*self.D['x0o1']
                self.w['x0o1'] += self.D['x0o1']

                self.D['h1o1'] = eta*self.G['h1o1']+alpha*self.D['h1o1']
                self.w['h1o1'] += self.D['h1o1']

                self.D['h2o1'] = eta * self.G['h2o1'] + alpha * self.D['h2o1']
                self.w['h2o1'] += self.D['h2o1']

                self.D['x0h1'] = eta * self.G['x0h1'] + alpha * self.D['x0h1']
                self.w['x0h1'] += self.D['x0h1']

                self.D['x1h1'] = eta * self.G['x1h1'] + alpha * self.D['x1h1']
                self.w['x1h1'] += self.D['x1h1']

                self.D['x2h1'] = eta * self.G['x2h1'] + alpha * self.D['x2h1']
                self.w['x2h1'] += self.D['x2h1']

                self.D['x0h2'] = eta * self.G['x0h2'] + alpha * self.D['x0h2']
                self.w['x0h2'] += self.D['x0h2']

                self.D['x1h2'] = eta * self.G['x1h2'] + alpha * self.D['x1h2']
                self.w['x1h2'] += self.D['x1h2']

                self.D['x2h2'] = eta * self.G['x2h2'] + alpha * self.D['x2h2']
                self.w['x2h2'] += self.D['x2h2']

        return epoca

if __name__ == "__main__":
    padroes = [[0, 0], [0, 1], [1, 0], [1, 1]]
    desejado = [0, 1, 1, 0]
    eta = 0.2
    alpha = 0.2

    b = Backpropagation()
    print(b.w)
    print('________SEM TREINAMENTO_______')
    print(' 0 - 0  = ' + repr(b.y([0, 0])))
    print(' 0 - 1  = ' + repr(b.y([0, 1])))
    print(' 1 - 0  = ' + repr(b.y([1, 0])))
    print(' 1 - 1  = ' + repr(b.y([1, 1])))
    print('___________TREINAMENTO________')
    e = b.treinar(padroes,desejado,eta,alpha)
    print(e)
    print('_____________RESULTADO________')
    print(' 0 - 0  = ' + repr(b.y([0, 0])))
    print(' 0 - 1  = ' + repr(b.y([0, 1])))
    print(' 1 - 0  = ' + repr(b.y([1, 0])))
    print(' 1 - 1  = ' + repr(b.y([1, 1])))
    print('_____________________________*')