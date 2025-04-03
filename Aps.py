import numpy as np
import matplotlib.pyplot as plt

# Este archivo contiene varias funciones que usamos en la materia APS

#Semana 1: Generador de senoidal
def generador_sen(vmax, dc, ff, ph, nn, fs):
    '''
    Esta funcion genera una señal senoidal.
    descripcion de los parametros:
    vmax:amplitud max de la senoidal [Volts]
    dc:valor medio [Volts]
    ff:frecuencia [Hz]
    ph:fase en [rad]
    nn:cantidad de muestras
    fs:frecuencia de muestreo [Hz]
    '''
    Ts= 1/fs
    tt=np.linspace(0,(nn-1)*Ts,nn)
    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc
    return tt, xx

#Semana 4: Simulacion de un ADC 
def adc_sim(xn, B, kn, Vf, N, fs, ymin, ymax, ymin_2):
    q = Vf/(2**(B))
    print('q:',q)
    
    pot_ruido_cuant = (q**2)/12
    pot_ruido_analog = pot_ruido_cuant*kn
    print('Potencia ruido cuant:',pot_ruido_cuant)
    print('Potencia ruido analog:',pot_ruido_analog)

    nn = np.random.normal(0,np.sqrt(pot_ruido_analog),N)
    sr = xn + nn
    srq = np.round(sr/q)*q
    nq = srq-sr

    tt = np.arange(N)/fs

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(tt, xn, 'purple', linestyle =':', label='$S$:Señal Analogica', linewidth=2)
    plt.plot(tt, sr, 'deeppink', label='$S_R$: S con ruido Analogico (ADC IN)', linewidth=2)
    plt.plot(tt, srq, 'coral',linestyle = '-', label='$S_{RQ}$: Señal cuantizada  (ADC OUT)', linewidth=2) 
    plt.plot(tt, nq, 'hotpink', label=f'$n_Q$: Ruido de cuantización (q={q:.3})', linewidth=1)
    plt.plot(tt, nn, 'blue', label=f'$n_n$: Ruido analogico', linewidth=1)
    plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Volts')
    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.7)
    plt.tight_layout()
    #plt.show()
    
    #plt.figure(figsize=(8,5))
    plt.subplot(1,2,2)
    bins=10
    plt.hist(nq.flatten()/q,bins=bins,color='purple',alpha=0.7)
    plt.plot([-0.5,-0.5,0.5,0.5],[0,N/bins,N/bins,0],'--r')
    plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
    plt.xlabel('Pasos quant (q)')
    plt.tight_layout()
    plt.show()


    df = fs/N
    ft_SR = 1/N*np.fft.fft(sr)
    ft_Srq = 1/N*np.fft.fft(srq)
    ft_As = 1/N*np.fft.fft(xn)
    ft_Nq = 1/N*np.fft.fft(nq)
    ft_Nn = 1/N*np.fft.fft(nn)
    ff = np.linspace(0,(N-1)*df,N)
    bfrec = ff <= fs/2

    Nnq_mean = np.mean(np.abs(ft_Nq[bfrec])**2)
    nNn_mean = np.mean(np.abs(ft_Nn[bfrec])**2)

    plt.figure(figsize=(14,8))
    plt.subplot(2,1,1)
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='purple', ls='-', label='$ s $ (sig.)', lw = 1 )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), 'deeppink',linestyle = '-', label='$ s_R $' , lw = 1)
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), 'coral',lw=2, label='$ s_{RQ} = Q_{B,V_F}\{s_R\}$' )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), ls='--',color = 'blue', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), ls='--',color ='hotpink', label= '$ \overline{n_n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), 'blue', label='$ n_Q $', lw = 1 ,ls=':')
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), 'hotpink', label='$ n_n $', lw = 1 , ls= ':')
    plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 1  )
    plt.title(f'Espectro ADC {B} bits - Vf={Vf}V')
    plt.ylabel('Densidad potencia [dB]')
    plt.xlabel('Freq [Hz]')
    plt.ylim(ymin,ymax)
    plt.legend(loc= 'upper right')
    plt.grid(True,linestyle='--',alpha=0.5)
    #plt.show()

    #plt.figure(figsize=(14,6))
    plt.subplot(2,1,2)
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='purple', ls='-', label='$ s $ (sig.)', lw = 1 )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), 'deeppink',linestyle = '-', label='$ s_R $' , lw = 1)
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), 'coral',lw=2, label='$ s_{RQ} = Q_{B,V_F}\{s_R\}$' )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), ls='--',color = 'blue', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), ls='--',color ='hotpink', label= '$ \overline{n_n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), 'blue', label='$ n_Q $', lw = 1 ,ls=':')
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), 'hotpink', label='$ n_n $', lw = 1 , ls= ':')
    plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 1  )
    plt.title(f'Espectro (zoom) ADC {B} bits - Vf={Vf}V')
    plt.ylabel('Densidad potencia [dB]')
    plt.xlabel('Freq [Hz]')
    plt.ylim(ymin_2,ymax)
    plt.xlim(0,10)
    plt.legend(loc='upper right')
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.tight_layout()
    plt.show()