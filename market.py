import numpy as np

#Class for a stock
class Stock:
    def __init__(self, mu, dt, s0, sigma):
        self.mu = mu #Mean
        self.dt = dt #Time step, 1/252 for daily data
        self.s0 = s0 #Initial stock price
        self.sigma = sigma #volatility
        self.current_price = s0 
    
    def calculate_next_price(self):
        z = np.random.normal(0,1) #Standard normal for GBM
        new_price =  self.current_price * np.exp((self.mu - (self.sigma ** 2) / 2)*self.dt + self.sigma*z*np.sqrt(self.dt))
        self.current_price = new_price
        
    def reset(self):
        self.current_price = self.s0
    
    def get_price(self):
        return self.current_price



#Class for a bond
class Bond():
    def __init__(self,b0,dt,r):
        self.b0 = b0 #Amount inveted in bond
        self.dt = dt #Time period held
        self.r = r #Yearly risk-free rate
        self.current_value = b0
    def calculate_next_price(self):
        new_value = self.b0*(1+self.r)**(1/self.dt)
        self.current_value = new_value
        
    def get_price(self):
        return self.current_value
    
    def reset(self):
        self.current_value = self.b0