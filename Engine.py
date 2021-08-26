import numpy as np
import pandas as pd
import scipy.optimize as opt
import pyswarm as ps
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class Well:

    def __init__(self, name):
        self.name = name

    def readdata(self, head_curve, hp_curve, eff_curve, pump_spec):
        self.pump_spec = pump_spec
        self.head_curve = head_curve
        self.hp_curve = hp_curve
        self.eff_curve = eff_curve
        
    def inputTubular(self, casing_id, tubing_id, casing_depth, tubing_depth):
        self.casing_id = casing_id
        self.tubing_id = tubing_id
        self.casing_depth = casing_depth
        self.tubing_depth = tubing_depth
        
    def getMaxMinRate(self, pumptype):
        perf = self.pump_spec
        raterange = perf['Range'].str.split('-',expand=True)
        perf['Minimum Rate'] = raterange[0].astype('int')
        perf['Maximum Rate'] = raterange[1].astype('int')
        pump_min_rate = perf[perf["Pump"] == pumptype]['Minimum Rate'].values[0]
        pump_max_rate = perf[perf["Pump"] == pumptype]['Maximum Rate'].values[0]
        return pump_min_rate, pump_max_rate
    
    def inputPump(self, pumptype, psd, stage, freq, headpump = 1):
        self.pumptype = pumptype
        self.pumpname = pumptype
        self.psd = psd
        self.stage = stage
        self.freq = freq
        self.head_factor = headpump
        pump_min_rate, pump_max_rate = self.getMaxMinRate(pumptype)
        self.min_rate = pump_min_rate
        self.max_rate = pump_max_rate
        self.mins_rate = pump_min_rate
        self.maxs_rate = pump_max_rate
        
    def inputCompletion(self, perfdepth, pres, tres, prodindex, iprtype='vogel', holdup=1.0):
        self.perfdepth = perfdepth
        self.sbhp = pres
        self.sbht = tres
        self.pi = prodindex
        self.iprtype = iprtype
        pbstar = ((self.gor*1000/self.sgg)**0.816)*((tres)**0.172)*(self.api**-0.989)
        self.pbub = 10**(1.7669 + 1.7447*np.log10(pbstar) - 0.30218*(np.log10(pbstar)**2))
        self.holdup_factor = holdup
    
    def inputPVT(self, api, sggas, sgwater, visco, viscw, gor, wc):
        self.api = api
        self.sgo = 141.5/(api + 131.5)
        self.sgg = sggas
        self.sgw = sgwater
        self.gor = gor
        self.watercut = wc
        self.visco = visco
        self.viscw = viscw
        self.tc = 168 + 325*self.sgg - 12.5*(self.sgg**2)
        self.pc = 677 + 15.0*self.sgg - 37.5*(self.sgg**2)
        
    def inputTestData(self, pbhp, rate, phf, thf, pdp, pip):
        self.pbhp_test = pbhp
        self.rate_test = rate
        self.phf = phf
        self.thf = thf
        self.pdp = pdp
        self.pip = pip
    
    def gasVisco(self, ppr,tpr,bv):
        a = [-2.46211820, 2.97054714, -2.86264054e-1,
            8.05420522e-3, 2.80860949, -3.49803305,
            3.60373020e-1, -1.044324e-2, -7.93385684e-1,
            1.39643306, -1.49144925e-1, 4.41015512e-3,
            8.39387178e-2, -1.86408848e-1, 2.03367881e-2,
            -6.09579263e-4]
        lnvisc = a[0] + a[1]*ppr + a[2]*(ppr**2) + a[3]*(ppr**3) + tpr*(a[4] + a[5]*ppr + a[6]*(ppr**2) + a[7]*(ppr**3)) + (tpr**2)*(a[8] + a[9]*ppr + a[10]*(ppr**2) + a[11]*(ppr**3)) + (tpr**3)*(a[12] + a[13]*ppr + a[14]*(ppr**2) + a[15]*(ppr**3))
        viscg = (np.exp(lnvisc)/tpr)*bv
        return viscg
    
    def baseVisco(self, temp, sg):
        result = (1.709e-5 - 2.062e-6*sg)*(temp) + 8.188e-3 - 6.15e-3*np.log10(sg)
        return result
    
    def zFactor (self, ppr,tpr):
        a = [0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210]
        rho = 0.27*ppr/tpr
        zfactor = (a[0] + a[1]/tpr + a[2]/(tpr**3) + a[3]/(tpr**4) + a[4]/(tpr**5))*rho + (a[5] + a[6]/tpr + a[7]/(tpr**2))*(rho**2) - a[8]*(a[6]/tpr + a[7]/(tpr**2))*(rho**5) + a[9]*(1+a[10]*(rho**2))*((rho**2)/(tpr**3))*np.exp(-1*a[10]*(rho**2)) + 1
        error = 999
        while (error > 0.0001):
            rho = 0.27*ppr/(zfactor*tpr)
            newzfactor = (a[0] + a[1]/tpr + a[2]/(tpr**3) + a[3]/(tpr**4) + a[4]/(tpr**5))*rho + (a[5] + a[6]/tpr + a[7]/(tpr**2))*(rho**2) - a[8]*(a[6]/tpr + a[7]/(tpr**2))*(rho**5) + a[9]*(1+a[10]*(rho**2))*((rho**2)/(tpr**3))*np.exp(-1*a[10]*(rho**2)) + 1
            error = abs((newzfactor - zfactor)/zfactor)
            zfactor = newzfactor
        return zfactor
    
    def chenFrictionFactor(self,nre):
        eps = 0.0006
        temp = -4*np.log10(eps/3.7065 - (5.0452/nre)*np.log10((eps**1.1098)/2.8257 + (7.149/nre)**0.8981))
        result = (1/temp)**2
        return result
    
    def getLinearIPR(self,kind='graph'):
        pi = self.pi
        pbhp = [i for i in range (0, int(self.sbhp), 1)]
        prate = [pi*(self.sbhp - i) for i in pbhp]
        self.iprdata = {
                'Pressure':pbhp,
                'Rate':prate
            }
        if kind == 'graph':
            chart = px.scatter(x=prate,y=pbhp,width=600,height=450)
            #chart.add_trace(go.Scatter(x=prate,y=pbhp,
                                #mode='lines',
                                #name='Line'))

            chart.update_layout(title={'text':'Linear IPR Plot : '+str(self.name),
                                        'xanchor' : 'left',
                                        'yanchor' :'top',
                                        'x' : 0},
                               xaxis_title="Rate (STB/d)",
                               yaxis_title="Pwf (psia)")
            st.plotly_chart(chart)
            #plt.figure(figsize=(10,6))
            #plt.plot(prate,pbhp)
            #plt.title(self.name + " IPR")
            #plt.xlabel("Rate (STB/d)")
            #plt.ylabel("Pwf (psia)")
            #plt.show()
        elif kind == 'data':
            return {
                'Pressure':pbhp,
                'Rate':prate
            }
        
    def getVogelIPR(self,kind='graph'):
        pi = self.pi
        pbhp = [i for i in range (0, int(self.sbhp), 1)]
        prate = []
        for press in pbhp:
            if press >= self.pbub:
                rate = pi*(self.sbhp - press)
            else:
                rate = (pi*(self.sbhp-self.pbub)+(pi*self.pbub/1.8)*(1- 0.2*(press/self.pbub) - 0.8*((press/self.pbub)**2)))
            prate.append(rate)
        self.iprdata = {
                'Pressure':pbhp,
                'Rate':prate
            }
        if kind == 'graph':
            chart = px.scatter(x=prate,y=pbhp,width=600,height=450)
            #chart.add_trace(go.Scatter(x=prate,y=pbhp,
                                #mode='lines',
                                #name='Line'))

            chart.update_layout(title={'text':'Vogel IPR Plot : '+str(self.name),
                                        'xanchor' : 'left',
                                        'yanchor' :'top',
                                        'x' : 0},
                               xaxis_title="Rate (STB/d)",
                               yaxis_title="Pwf (psia)")
            st.plotly_chart(chart)
            #plt.figure(figsize=(10,6))
            #plt.plot(prate,pbhp)
            #plt.title(self.name + " IPR")
            #plt.xlabel("Rate (STB/d)")
            #plt.ylabel("Pwf (psia)")
            #plt.show()
        elif kind == 'data':
            return {
                'Pressure':pbhp,
                'Rate':prate
            }
    
    def pumpPerformance(self, valrate, pump, freq):
        valrate = valrate*60/self.freq
        head = self.head_curve
        hp = self.hp_curve
        eff = self.eff_curve
        try:
            if (head[(head["rate"] == valrate)][pump].count() == 0):
                minrate = head[head["rate"] < valrate]["rate"].iloc[-1]
                minidx = head[head["rate"] == minrate].index[0]
                maxrate = head[head["rate"] > valrate]["rate"].iloc[0]
                maxidx = head[head["rate"] == maxrate].index[0]
                minhead = head[pump][minidx]
                maxhead = head[pump][maxidx]
                minhp = hp[pump][minidx]
                maxhp = hp[pump][maxidx]
                mineff = eff[pump][minidx]
                maxeff = eff[pump][maxidx]
                valhead = (valrate - minrate)*(maxhead - minhead)/(maxrate - minrate) + minhead
                valeff = (valrate - minrate)*(maxeff - mineff)/(maxrate - minrate) + mineff
                valhp = (valrate - minrate)*(maxhp - minhp)/(maxrate - minrate) + minhp
            else:
                valhead = head[head['rate'] == valrate][pump].values[0]
                valeff = eff[eff['rate'] == valrate][pump].values[0]
                valhp = hp[hp['rate'] == valrate][pump].values[0]
            return valhead,valeff,valhp
        except:
            return 0,0,0
    
    def getPressureProfile(self, rate, kind='graph'):
        if (rate == 0):
            rate = 1.1
        head_factor = self.head_factor
        freq=self.freq
        phf=self.phf
        thf=self.thf
        watercut = self.watercut/100
        qg = rate*(1-watercut)*self.gor
        glr = qg*1000/rate
        tubing_area = 0.25*3.14159*((self.tubing_id/12)**2)
        casing_area = 0.25*3.14159*((self.casing_id/12)**2)
        qo = rate*(1-watercut)
        qw = rate*watercut
        sgl = (qo*self.sgo + qw*self.sgw)/rate
        viscl = (qo*self.visco + qw*self.viscw)/rate
        mt = sgl*62.4*rate*5.615 + 0.0765*self.sgg*qg*1000
        n = 30
        delta = self.tubing_depth/29
        pressure = [0]*n
        temperature = [0]*n
        depth = [0]*n
        press = phf
        temp = thf
        ppr = press/self.pc
        tpr = (temp + 460)/self.tc
        dep = 0
        ift = 30
        pressure[0] = press
        temperature[0] = temp
        depth[0] = dep
        dpdz = []
        for i in range (1,30):
            viscg = self.gasVisco(ppr, tpr, self.baseVisco(temp, self.sgg))
            zfactor = self.zFactor(ppr,tpr)
            usg = (qg*1000*zfactor*((temp + 460)/520)*(14.7/press)/86400)/tubing_area
            usl = (rate*5.615/86400)/tubing_area
            um = usl + usg
            rhol = sgl*62.4
            nvl = 1.938*usl*((rhol/ift)**0.25)
            nvg = 1.938*usg*((rhol/ift)**0.25)
            ndim = 120.872*(self.tubing_id/12)*np.sqrt(rhol/ift)
            nliq = 0.15726*viscl*((1/(rhol*(ift**3)))**0.25)
            #Log(NL + 3) (book) or Log(NL) + 3 (excel) ???
            x1 = np.log10(nliq + 3)
            y = -2.69851 + 0.15840954*x1 - 0.55099756*(x1**2) + 0.54784917*(x1**3) - 0.12194578*(x1**4)
            cnl = 10**y
            x2 = (nvl*(press**0.1)*cnl)/((nvg**0.575)*(14.7**0.1)*ndim)
            yl_per_psi = -0.10306578 + 0.617774*(np.log10(x2)+6)-0.632946*((np.log10(x2)+6)**2)+0.29598*((np.log10(x2)+6)**3)-0.0401*((np.log10(x2)+6)**4)
            x3 = (nvg*(nliq**0.38))/(ndim**2.14)
            if (x3 < 0.012):
                x3 = 0.012
            psi = 0.91163 - 4.82176*x3 + 1232.25*(x3**2) - 22253.6*(x3**3) + 116174.3*(x3**4)
            yl = psi * yl_per_psi
            yl = self.holdup_factor*yl + (1-self.holdup_factor)*(yl**2)
            nre = 0.022*mt/(self.tubing_id*(viscl**yl)*(viscg**(1-yl)))
            ffactor = self.chenFrictionFactor(nre)
            rhog = 28.97*self.sgg*press/(zfactor*10.73*(temp + 460))
            rhoavg = rhol*yl + rhog*(1 - yl)
            dp_dz = (1/144)*(rhoavg + (ffactor*(mt**2))/(7.413e10*((self.tubing_id/12)**5)*rhoavg))
            dep = dep + delta
            press = press + delta*dp_dz
            temp = thf + ((self.sbht - thf)/self.tubing_depth)*dep
            ppr = press/self.pc
            tpr = (temp + 460)/self.tc
            pressure[i] = press
            depth[i] = dep
            temperature[i] = temp
            dpdz.append(dp_dz)
            
        head,eff,hp = self.pumpPerformance(rate, self.pumptype, freq)
        dp_across_pump = head*self.stage*0.433*head_factor
        press = press - dp_across_pump
        pressure.append(press)
        depth.append(dep)
        
        dp_casing = sgl*0.433*(self.perfdepth - self.psd)
        press = press + dp_casing
        self.pwf = press
        pressure.append(press)
        depth.append(self.perfdepth)
        
        pressure.append(self.sbhp)
        depth.append(self.perfdepth)
        if kind == 'graph':
            chart = px.scatter(x=pressure,y=depth,width=600,height=450)
            chart.add_trace(go.Scatter(x=pressure,y=depth,
                                mode='lines',
                                name='Line'))
            chart.update_yaxes(autorange="reversed")

            chart.update_layout(title={'text':'Pressure - Depth Profile : '+str(self.name),
                                        'xanchor' : 'left',
                                        'yanchor' :'top',
                                        'x' : 0},
                               xaxis_title="Pressure (psia)",
                               yaxis_title="Depth (ft)")
            st.plotly_chart(chart)
            #plt.figure(figsize=(10,6))
            #plt.plot(pressure,depth)
            #plt.gca().invert_yaxis()
            #plt.title(self.name + " Pressure - Depth Profile")
            #plt.xlabel('Pressure (psia)')
            #plt.ylabel('Depth (ft)')
            #plt.show()
        elif kind == 'data':
            return {
                'Pressure':pressure,
                'Depth':depth
            }
    def getHagedornBrownTPR(self, kind='graph'):
        prate = np.array([i for i in range(int(min(self.iprdata['Rate'])),int(max(self.iprdata['Rate'])),100)])
        pbhp = np.array([self.getPressureProfile(rate=i, kind='data')["Pressure"][-2] for i in prate])
        self.tprdata = {
                'Pressure':pbhp,
                'Rate':prate
            }
        if kind == 'graph':
            plt.figure(figsize=(10,6))
            plt.plot(prate,pbhp)
            plt.title(self.name + " TPR")
            plt.xlabel("Rate (STB/d)")
            plt.ylabel("Pwf (psia)")
            plt.show()
        elif kind == 'data':
            return {
                'Pressure':pbhp,
                'Rate':prate
            }
    
    def calculateBHPIPR(self, rate):
        ipr = self.iprdata
        return np.interp(rate, sorted(self.iprdata['Rate'], reverse=False), sorted(self.iprdata['Pressure'], reverse=True))
        
    def calculateBHPTPR(self, rate):
        tpr = self.tprdata
        return np.interp(rate, tpr['Rate'], tpr['Pressure'])
    
    def getNodal(self, kind='graph'):
        ipr = self.iprtype
        if (ipr == 'vogel'):
            iprdata = self.getVogelIPR(kind='data')
        elif (ipr == 'linear'):
            iprdata = self.getLinearIPR(kind='data')
        tprdata = self.getHagedornBrownTPR(kind='data')
        if kind == 'graph':
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=iprdata['Rate'],y=iprdata['Pressure'],
                                mode='lines',
                                name='IPR'))
            
            chart.add_trace(go.Scatter(x=tprdata['Rate'],y=tprdata['Pressure'],
                                mode='lines',
                                name='TPR'))

            chart.update_layout(title={'text':'Nodal Plot : '+str(self.name),
                                        'xanchor' : 'left',
                                        'yanchor' :'top',
                                        'x' : 0},
                               autosize=False,
                               width=600,
                               height=450,
                               xaxis_title="Rate (STB/d)",
                               yaxis_title="Pwf (psia)")
            st.plotly_chart(chart)
            #plt.figure(figsize=(10,6))
            #plt.title(self.name + " Nodal Plot")
            #plt.xlabel("Rate (STB/d)")
            #plt.ylabel("Pwf (psia)")
            #plt.plot(iprdata['Rate'],iprdata['Pressure'], label='IPR')
            #plt.plot(tprdata['Rate'],tprdata['Pressure'], label='TPR')
            #plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
            #plt.show()
        elif kind == 'data':
            return iprdata,tprdata
    
    def getOperatingPoint(self):
        if (max(self.iprdata['Rate']) > max(self.tprdata['Rate'])):
            max_rate = max(self.tprdata['Rate'])
        else:
            max_rate = max(self.iprdata['Rate'])
        if (min(self.iprdata['Rate']) < min(self.tprdata['Rate'])):
            min_rate = min(self.tprdata['Rate'])
        else:
            min_rate = min(self.iprdata['Rate'])
        raterange = np.array([i for i in range (int(min_rate), int(max_rate),100)])
        iprpress = np.array([self.calculateBHPIPR(i) for i in raterange])
        tprpress = np.array([self.calculateBHPTPR(i) for i in raterange])
        diff = iprpress - tprpress
        tanda = np.sign(diff)
        start = tanda[0]
        end = tanda[0]
        for i in range (len(tanda)):
            if (tanda[i] != start):
                end = end*-1
                idx = i
                break
        if (start == end):
            rate = np.nan
            pressure = np.nan
        else:
            x1 = raterange[idx-1]
            y1 = tprpress[idx-1]
            x2 = raterange[idx]
            y2 = tprpress[idx]
            y3 = iprpress[idx-1]
            y4 = iprpress[idx]
            m1 = (y4-y3)/(x2-x1)
            m2 = (y2-y1)/(x2-x1)
            rate = ((y1-y3)+(m1*x1 - m2*x1))/(m1-m2)
            pressure = m1*rate + (y3 -m1*x1)
        return rate, pressure
        
    def getOperatingPointRes(self):
        if (max(self.iprdata['Rate']) > max(self.tprdata['Rate'])):
            max_rate = max(self.tprdata['Rate'])
        else:
            max_rate = max(self.iprdata['Rate'])
        if (min(self.iprdata['Rate']) < min(self.tprdata['Rate'])):
            min_rate = min(self.tprdata['Rate'])
        else:
            min_rate = min(self.iprdata['Rate'])
        raterange = np.array([i for i in range (int(min_rate), int(max_rate),100)])
        iprpress = np.array([self.calculateBHPIPR(i) for i in raterange])
        tprpress = np.array([self.calculateBHPTPR(i) for i in raterange])
        diff = iprpress - tprpress
        tanda = np.sign(diff)
        start = tanda[0]
        end = tanda[0]
        for i in range (len(tanda)):
            if (tanda[i] != start):
                end = end*-1
                idx = i
                break
        if (start == end):
            rate = np.nan
            pressure = np.nan
        else:
            x1 = raterange[idx-1]
            y1 = tprpress[idx-1]
            x2 = raterange[idx]
            y2 = tprpress[idx]
            y3 = iprpress[idx-1]
            y4 = iprpress[idx]
            m1 = (y4-y3)/(x2-x1)
            m2 = (y2-y1)/(x2-x1)
            rate = ((y1-y3)+(m1*x1 - m2*x1))/(m1-m2)
        pressure = m1*rate + (y3 -m1*x1)
        st.info('''**Operating Point**
                    
        Operating Rate (STB/d): {}
        Operating Pressure (psia): {}
        '''.format(rate ,pressure))
        return rate, pressure
    
    def getIPRTPRData(self):
        ipr, tpr = self.getNodal(kind='data')
        self.iprdata = ipr
        self.tprdata = tpr
    
    def constraintPI(self, x):
        return x[0]
    
    def constraintMinHF(self, x):
        return x[0]
    
    def constraintMaxHF(self, x):
        return 2 - x[0]
    
    def constraintMinHead(self, x):
        return x[0]
    
    def constraintMaxHead(self, x):
        return 1 - x[0]
    
    def matchIPR(self):
        print('Matching ...')
        x = np.array([self.pi])
        cons = [{'type':'ineq', 'fun':self.constraintPI}]
        result = opt.minimize(self.lossIPR, x, method = 'SLSQP', constraints=cons, tol=0.0005)
        self.pi = result['x'][0]
        st.write('**Matching IPR** : {} '.format(result))
        st.info('**Matched PI Result** : {} '.format(result['x'][0]))
        return result
    
    def lossIPR(self,x):
        self.pi = x[0]
        print('PI : ',round(self.pi,3))
        if (self.iprtype == 'vogel'):
            ipr = self.getVogelIPR(kind='data')
        elif (self.iprtype == 'linear'):
            ipr = self.getLinearIPR(kind='data')
        iprcoef = np.polyfit(ipr["Rate"], ipr["Pressure"], 3)
        pbhp_calc = iprcoef[0]*(self.rate_test**3) + iprcoef[1]*(self.rate_test**2) + iprcoef[2]*(self.rate_test) + iprcoef[3]
        loss = abs((pbhp_calc - self.pbhp_test)/pbhp_calc)*100
        return loss
        
    def matchFlowCorrelation(self):
        print('Matching ...')
        x = np.array([self.holdup_factor])
        cons = [{'type':'ineq', 'fun':self.constraintMinHF},{'type':'ineq', 'fun':self.constraintMaxHF}]
        result = opt.minimize(self.lossFlowCorrelation, x, method = 'SLSQP', constraints=cons, tol=0.0005)
        self.holdup_factor = result['x'][0]
        st.write('**Matching Flow Correlation** : {} '.format(result))
        st.info('**Matched Holdup Factor Result** : {} '.format(result['x'][0]))
        return result
    
    def lossFlowCorrelation(self,x):
        self.holdup_factor = x[0]
        print('Holdup factor : ',round(self.holdup_factor,3))
        pdp_calc = self.getPressureProfile(rate=self.rate_test, kind='data')['Pressure'][29]
        loss = abs((pdp_calc - self.pdp)/pdp_calc)*100
        return loss
    
    def matchPump(self):
        print('Matching ...')
        x = np.array([self.head_factor])
        cons = [{'type':'ineq', 'fun':self.constraintMinHead},{'type':'ineq', 'fun':self.constraintMaxHead}]
        result = opt.minimize(self.lossPump, x, method = 'SLSQP', constraints=cons, tol=0.0005)
        self.head_factor = result['x'][0]
        st.write('**Matching Pump** : {} '.format(result))
        st.info('**Matched Head Factor Result** : {} '.format(result['x'][0]))
        return result
    
    def lossPump(self,x):
        self.head_factor = x[0]
        print('Head factor : ',round(self.head_factor,3))
        pip_calc = self.getPressureProfile(rate=self.rate_test, kind='data')['Pressure'][30]
        loss = abs((pip_calc - self.pip)/pip_calc)*100
        return loss
    
    def getRate(self, x):
        self.phf = x[0]
        self.freq = x[1]
        self.stage = x[2]
        self.getIPRTPRData()
        rate, pressure = self.getOperatingPoint()
        self.iter += 1
        if (str(rate) == 'nan'):
            print('i:{:<3}, Phf:{:<5} psia, Frequency:{:<5} Hz, Stage:{:<5}, Rate:{:<6}, Power:{:<5} HP'.format(self.iter, round(self.phf,1), round(self.freq,1), round(self.stage),"Unconverged", round(self.getPower(rate),1)))
        else:
            print('i:{:<3}, Phf:{:<5} psia, Frequency:{:<5} Hz, Stage:{:<5}, Rate:{:<6} BBD, Power:{:<5} HP'.format(self.iter, round(self.phf,1), round(self.freq,1),round(self.stage),round(rate,1), round(self.getPower(rate),1)))
            # print('Iter : ',self.iter,'Phf : ', round(self.phf,1), ' Freq : ', round(self.freq,1), ' Rate : ', , 'Power : ', round(self.getPower(rate),1))
        return -rate
    
    def getPower(self, rate):
        head, eff, hp = self.pumpPerformance(rate, self.pumptype, self.freq)
        return hp*self.stage
    
    def consPower(self, x):
        self.phf = x[0]
        self.freq = x[1]
        self.stage = x[2]
        self.getIPRTPRData()
        rate, pressure = self.getOperatingPoint()
        head, eff, hp = self.pumpPerformance(rate, self.pumptype, self.freq)
        power = self.stage*hp
        return [self.maxpower - power]
    
    def maximizeRate(self, phf_range, freq_range, stage_range, az, bz, cz, dz):
        lb = [phf_range[0], freq_range[0], stage_range[0]]
        ub = [phf_range[1], freq_range[1], stage_range[1]]
        self.iter = 0
        print('Maximizing...')
        xopt, fopt = ps.pso(self.getRate, lb, ub, f_ieqcons=self.consPower, swarmsize=az, maxiter=bz, minstep=cz, minfunc=dz)
        st.info('''**Optimized Parameters**
        
        Optimized Phf (psia): {}
        Optimized Frequency (Hz): {}
        Optimized Stage: {}
        Maximum Rate (BBD): {}
        '''.format(xopt[0], xopt[1], xopt[2], -1*fopt))
        return (xopt,fopt)

st.title('**Production Optimization**')
st.markdown('''
This program is able to **Create an automatic well model** and match the model to the actual well data then **Find the optimum parameter** of the well based on well constraints.
* First, you need to either input your own dataset or use sample dataset
* Then, you need to select the method that used in model
* Last, you need to input the value for optimization
''')

st.sidebar.title('**Input Data Section**')

input1 = st.sidebar.selectbox('Choose Well Dataset: ',['Your Dataset','Sample Dataset'])
if input1 == 'Your Dataset':
    upload1 = st.sidebar.file_uploader("Drop your Well Dataset (.xlsx): ")
    if upload1 is not None:
        well_data = pd.read_excel(upload1)
    else:
        well_data = pd.read_excel('Fix_Well_Data.xlsx')

else:
    well_data = pd.read_excel('Fix_Well_Data.xlsx')
    
input2 = st.sidebar.selectbox('Choose pump-performance Dataset: ',['Your Dataset','Sample Dataset'])
if input2 == 'Your Dataset':
    upload2 = st.sidebar.file_uploader("Drop your pump-performance Datase (.xlsx): ")
    if upload2 is not None:
        headcurve = pd.read_excel(upload2,sheet_name='head')
        hpcurve = pd.read_excel(upload2,sheet_name='hp')
        effcurve = pd.read_excel(upload2,sheet_name='eff')
    else:
        headcurve = pd.read_excel('pump-performance.xlsx',sheet_name='head')
        hpcurve = pd.read_excel('pump-performance.xlsx',sheet_name='hp')
        effcurve = pd.read_excel('pump-performance.xlsx',sheet_name='eff')

else:
    headcurve = pd.read_excel('pump-performance.xlsx',sheet_name='head')
    hpcurve = pd.read_excel('pump-performance.xlsx',sheet_name='hp')
    effcurve = pd.read_excel('pump-performance.xlsx',sheet_name='eff')

input3 = st.sidebar.selectbox('Choose PumpSpecData Dataset: ',['Your Dataset','Sample Dataset'])
if input3 == 'Your Dataset':
    upload3 = st.sidebar.file_uploader("Drop your PumpSpecData Dataset (.csv): ")
    if upload3 is not None:
        pumpspec = pd.read_csv(upload3)
    else:
        pumpspec = pd.read_csv('PumpSpecData.csv')

else:
    pumpspec = pd.read_csv('PumpSpecData.csv')
 
st.sidebar.subheader('Input Section')
well_selection = st.sidebar.selectbox('Well Selection', well_data['Well Only'].values)
iprmodel= st.sidebar.selectbox('IPR Type', ['linear', 'vogel'])
tprmodel= st.sidebar.selectbox('TPR Type', ['Hagedorn Brown'])

st.sidebar.subheader('Optimization Method')
Optimizationmethod = st.sidebar.selectbox('Optimization Method', ['PSO (Particle Swarm Optimization'])
aa = st.sidebar.number_input(label="Swarm Size", format="%f",value=10.)
bb = st.sidebar.number_input(label="Maximum Iteration", format="%f",value=100.)
cc = st.sidebar.number_input(label="Minimum Step", format="%f",value=1.)
dd = st.sidebar.number_input(label="Minimum func", format="%f",value=1.)

st.sidebar.subheader('Maximize Rate')
phflow = int(st.sidebar.slider('Phf Lower Boundary ', min_value=0, max_value=2000,value=300))
phfup = int(st.sidebar.slider('Phf Upper Boundary ', min_value=0, max_value=2000,value=800))
freqlow = int(st.sidebar.slider('Frequency Lower Boundary ', min_value=0, max_value=500,value=30))
frequp = int(st.sidebar.slider('Frequency Upper Boundary ', min_value=0, max_value=500,value=90))
stagelow = float(st.sidebar.slider('Stage Lower Boundary ', min_value=0., max_value=10.0,value=0.5))
stageup = float(st.sidebar.slider('Stage Upper Boundary ', min_value=0., max_value=10.0,value=1.5))

st.sidebar.subheader('Tubular Input')
casingid = st.sidebar.number_input(label="Input casing_id", format="%f",value=9.)

st.sidebar.subheader('PVT Input')
apival = st.sidebar.number_input(label="Input deg API", format="%f",value=35.)
sggasval = st.sidebar.number_input(label="Input SG Gas", format="%f",value=0.7)
sgwaterval = st.sidebar.number_input(label="Input SG Water", format="%f",value=1.02)
viscoval = st.sidebar.number_input(label="Input Viscosity of Oil", format="%f",value=2.)
viscwval = st.sidebar.number_input(label="Input Viscosity of Water", format="%f",value=0.5)

st.sidebar.subheader('Completion Input')
prodindexval = st.sidebar.number_input(label="Input Productivity Index", format="%f",value=10.)
holdupval = st.sidebar.number_input(label="Input Holdup Factor", format="%f",value=1.)

st.sidebar.subheader('Pump Input')
headpumpval = st.sidebar.number_input(label="Input Head Factor", format="%f",value=1.)

well_count = well_data[well_data['Well Only']==well_selection].index[0]
well = Well(well_data['Well Only'].values[well_count])
well.readdata(head_curve=headcurve, hp_curve=hpcurve, eff_curve=effcurve, pump_spec=pumpspec)
well.inputTubular(casing_id=casingid, tubing_id=well_data['TUBING ID (INCH)'].values[well_count], casing_depth=well_data['BOT. INTVAL'].values[well_count], tubing_depth=well_data['TOP INTVAL'].values[well_count])
well.inputPump(pumptype=well_data['PUMP'].values[well_count], psd=well_data['PSD'].values[well_count], stage=well_data['STGS'].values[well_count], freq=well_data['HZ'].values[well_count], headpump = headpumpval)
well.inputPVT(api=apival, sggas=sggasval, sgwater=sgwaterval, visco=viscoval, viscw=viscwval, gor=well_data['GOR'].values[well_count], wc=well_data['BS&W'].values[well_count])
well.inputCompletion(perfdepth=(well_data['BOT. INTVAL'].values[well_count] + well_data['TOP INTVAL'].values[well_count])/2, pres=well_data['SBHP (PSI)'].values[well_count], tres=well_data['SBHT (F)'].values[well_count], prodindex=prodindexval, iprtype=iprmodel, holdup=holdupval)
well.inputTestData(pbhp=well_data['PBHP_x'].values[well_count], rate=well_data['BFPD_x'].values[well_count], phf=well_data['FTHP'].values[well_count], thf=well_data['FLT'].values[well_count], pdp=well_data['PDP'].values[well_count], pip=well_data['PIP'].values[well_count])

st.title('**IPR Profile**')
if iprmodel == 'linear':
    well.getLinearIPR()
elif iprmodel == 'vogel':
    well.getVogelIPR()
    
well.matchIPR()

st.write('**IPR after PI matched**')
if iprmodel == 'linear':
    well.getLinearIPR()
elif iprmodel == 'vogel':
    well.getVogelIPR()

st.title('**Pressure Profile**')

well.getPressureProfile(well.rate_test)
well.matchFlowCorrelation()
st.write('**Pressure Profile after Holdup Factor matched**')
well.getPressureProfile(well.rate_test)

well.matchPump()
st.write('**Pressure Profile after Holdup and Head Factor matched**')
well.getPressureProfile(well.rate_test)

st.title('**Nodal Analysis**')
well.getNodal()
well.getOperatingPointRes()


st.title('**Maximize Rate**')
well.maxpower = well_data['HP'].values[well_count]
well.maximizeRate(phf_range=(phflow,phfup), freq_range=(freqlow,frequp), stage_range=(int(well.stage*float(stagelow)),int(well.stage*float(stageup))),az=int(aa), bz=int(bb), cz=int(cc), dz=int(dd))

check_data = st.checkbox('Display Input Dataset')

if check_data:
    st.write("**Well Data**")
    st.write(well_data)
    st.write("**Pump Spec**")
    st.write(pumpspec)

check_data2 = st.checkbox('Display Pump Perfomance')

if check_data2:
    st.write("**Head Curve**")
    st.write(headcurve)
    st.write("**HP Curve**")
    st.write(pumpspec) 
    st.write("**Eff Curve**")
    st.write(effcurve)     
