#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
sum_am = 9
AM=2
"ΟΡΙΣΜΟΣ ΣΗΜΑΤΟΣ"
fm=1000*sum_am 
t = np.linspace(0, 4/fm, 500) 
y=np.cos(2*np.pi*fm*t)*np.cos(16*np.pi*fm*t) 
plt.figure(figsize=(12,9))
plt.figure(1)
plt.plot(t, y) #Αναπαράσταση του sq_triangle

#plt.savefig('export_images/starting_signal.png')


"1ο ΕΡΩΤΗΜΑ - (α)"
"(i)"
samples_fs1 = 20*4 # 25 δείγματα ανα περίοδο ( επι 4 αφού έχουμε 4 περι΄όδους )
t1 = np.linspace(0, 4/fm, samples_fs1) # δημ. του άξονα χρόνου για την πρώτη δειγματοληψία
y_fs1 = signal.resample(y,samples_fs1) # πρώτο δειγματοληπτημένο σήμα
plt.figure(figsize=(12,9))
plt.figure(2)
plt.stem(t1, y_fs1,':',basefmt=" ",use_line_collection=True) # Αναπαράσταση πρώτης δειγματοληψίας
#plt.savefig('export_images/1st_i.png')

"(ii)"
samples_fs2 = 100*4 # 60 δείγματα ανα περίοδο ( επι 4 αφού έχουμε 4 περι΄όδους )
t2 = np.linspace(0, 4/fm, samples_fs2) # δημ. του άξονα χρόνου για την δεύτερη δειγματοληψία
y_fs2 = signal.resample(y,samples_fs2) # δεύτερο δειγματοληπτημένο σήμα
plt.figure(figsize=(12,9))
plt.figure(3)
plt.stem(t2, y_fs2,':','g',basefmt=" ",markerfmt='go',use_line_collection=True) # Αναπαράσταση δεύτερης δειγματοληψίας

#plt.savefig('export_images/1st_ii.png')
"(iii)"
plt.figure(figsize=(12,9))
plt.figure(4)
plt.stem(t1,y_fs1,':','b',basefmt=" ",markerfmt='bo',use_line_collection=True) # Αναπαρ. και των δύο δειγματοληψιών
plt.stem(t2,y_fs2,':','g',basefmt=" ",markerfmt='go',use_line_collection=True)

#plt.savefig('export_images/1st_iii.png')

"1ο ΕΡΩΤΗΜΑ - (b)"
samples_fs3 = 5*4 # 5 δείγματα ανα περίοδο ( επι 4 αφού έχουμε 4 περι΄όδους )
t3 = np.linspace(0, 4/fm, samples_fs3) # δημ. του άξονα χρόνου για την τριτη δειγματοληψία
y_fs3 = signal.resample(y,samples_fs3) # τρίτο δειγματοληπτημένο σήμα
plt.figure(figsize=(12,9))
plt.figure(5)
# Αναπαράσταση τρίτης δειγματοληψίας
plt.stem(t3, y_fs3,':','g',basefmt=" ",markerfmt='go',use_line_collection=True) # Αναπαράσταση δεύτερης δειγματοληψίας

"ΔΕΥΤΕΡΟ ΕΡΩΤΗΜΑ @ΚΒΑΝΤΙΣΤΗΣ"
t = np.linspace(0, 1/fm, 500) 
y=np.cos(2*np.pi*fm*t)*np.cos(16*np.pi*fm*t) 
samples_fs1_y = 20 # 45 δείγματα
time = np.linspace(0, 1/fm, samples_fs1_y) # δημ. του άξονα χρόνου για την δειγματοληψία
fs1_y = signal.resample(y,samples_fs1_y) # πρώτο δειγματοληπτημένο σήμα
A1 = 1
# Αν fm άρτιο n = 4 αλλίως n = 5 (κβάντιση με 4 bits και 5 bits αντίστοιχα)
if sum_am % 2 == 0: 
    bits=4 
else: 
    bits=5        

#ορίσματα της εξίσωσης mid riser κβαντιστή

Levels = 2**bits    
d=1/(Levels) # βήματα κβάντισης
signal=d*np.floor(fs1_y/d) + d/2
#εξίσωση mid-riser κβαντιστή, χρήση της floor ώστε κατα τον κβαντισμό τα 
# σημεία της συνάρτησης να φτάνουν στο πλησιέστερο επίπεδο.
#==============================================================================
# quant_signal=d*np.floor(fs1_y/d) + d/2
#==============================================================================

#Κώδικας gray
def gray_code(n):
    if n <= 0:
        return []
    if n == 1:
        return ['0', '1']
    res = gray_code(n-1)
    return ['0'+s for s in res] + ['1'+s for s in res[::-1]]
#Κάλεσμα συνάρτησης gray με το num
g = gray_code(bits)

plt.figure(figsize=(12,9)) # κβαντισμένο δειγματοληπτημένο σήμα με fs1 
plt.figure(6)
plt.grid()
plt.yticks(np.arange(0.0,4.0+d/2,d),g[0:Levels]) 
plt.stem(time,signal,':',basefmt=" ",use_line_collection=True);

y_err = signal - fs1_y
y_err_10 = y_err[0:10]

#Τυπική Απόκλιση
variance_10=np.var(y_err_10)
st_deviation=math.sqrt(variance_10)
print('10 samples: \n' ,st_deviation)

y_err_20 = y_err[0:20]
variance_20=np.var(y_err_20)
st_deviation=math.sqrt(variance_20)
print('20 samples: \n' ,st_deviation)

#SNR για το (i)
SNR1 = 10*np.log10((np.var(fs1_y[0:10])/variance_10))
print('SNR1 = %f in dB' %SNR1)
#SNR για το (ii)
SNR2 = 10*np.log10((np.var(fs1_y[0:20])/variance_20)) 
print('SNR2 = %f in dB' %SNR2)
#Θεωρητικό SNR
var_th = 1./3*((A1**2)*(2**(-2*bits)))
av_power = (1./len(fs1_y)*sum(fs1_y**2))
SNR_Th = 10*np.log10(av_power/var_th)

print('SNR Theoretical = %f in dB' %SNR_Th)




#BIPOLAR RZ

#Συνένωση του g που περιέχει το gray code
c=''.join(g)
#Χωρίζουμε όλα τα bits που ενώσαμε πιο πάνω σε μια λίστα
bitstream=list(c)
#Το πλάτος με βάση το AM 
Amplitude = sum_am

#Δημιουργία των σημείων και του linspace ώστε να παρουσιάσουμε 
#την ροή μετάδοσης απο bits

#Αρχικοποίηση των counters που θα χρησιμοποιηθούν στο loop
i=0
counter = 0
#Αρχικοποίηση του array που θα περιέχει μέσα τα σημεία του σήματος
#όπου ο πίνακας είναι όσες θέσεις είναι το length του bitstream πίνακα
length = len(bitstream)
bitstream_signal=np.empty(length)

#Δημιουργία του πίνακα του χρόνου με βήμα 1ms, ξεκινά απο 0 και φτάνει μέχρι το 
# length/1000 ώστε να είναι σε milisecond
t = np.arange(0.0,length/1000,0.001)
#Αφού έχουμε bipolar rz πρέπει κάθε μηδενικό να εμφανίζεται ώς 0 και
#οι άσσοι να είναι εναλλάξ Amplitude και -Amplitude οπότε υλοποιείται ώς εξής,
#ώστε ο πίνακας bitstream_signal να γεμίσει με την σωστή πληροφορία:
for a in bitstream :
    if(a == '0'):
        bitstream_signal[i]=0
    else:
        counter += 1
        if(counter % 2 == 1):
            bitstream_signal[i] = Amplitude
        else:
            bitstream_signal[i] = -Amplitude
        
    i+=1

plt.figure(figsize=(25,12))
plt.figure(7)    
plt.grid()
plt.step(t,bitstream_signal)
np.arange(0.0,length/1000,0.001)
#plt.savefig('export_images/2nd_c.png')


# In[2]:


import numpy  as np
import matplotlib.pyplot as plt
import random
import math
from scipy.special import erfc

sum_am = 4
#αριθμός bits
bits = 46
#αριθμός samples για το  πρώτο  ερώτημα
samples = 1
#Χρ΄όνος κάθε bit
Tb = 0.5
# 1o Eρώτημα (a)
Amplitude = sum_am
sequence = np.empty(bits,dtype = int) #δημιουργία πίνακα 46 θέσεων
#σε κάθε θέση του πίνακα sequence θα μπεί 0 ή 1
for i in range (bits):
    sequence[i] = random.randint(0,1)
print(sequence)

#Δημιουργία πίνακα για το BPAM , όπου θα είναι bits*samples θέσεων έτσι ώστε
#για κάθε bit na υπάρχουν τόσα samples
BPAM = np.empty(bits*samples,dtype = int)
#αρχικοποίηση μεταβλητών που θα χρησιμοποιηθούν στα loops
lowbound = 0
upperbound = samples
#Για κάθε 0 το πλάτος είναι -Αmplitude για κάθε 1 το πλάτος είναι +Amplitude
for a in sequence :
    if(a == 0):
        for j in range (lowbound,upperbound):
            BPAM[j]= -Amplitude
    else:
        for j in range (lowbound,upperbound):
            BPAM[j] = Amplitude
    lowbound+=samples
    upperbound+=samples

#Δημιουργία άξονα χρόνου ώστε να παραστήσουμε την γραφική μας
t = np.linspace(0,bits*Tb,bits*samples)
plt.figure(figsize=(11,9))
plt.axis ([-5, 25, -5, 5])
plt.figure(1)
plt.step(t, BPAM, label = 'BPAM ')
plt.grid() 

E = Amplitude**2*Tb
o2=np.zeros(2)										
o1=np.zeros(2)
o1[0]=-math.sqrt(E)
o1[1]=math.sqrt(E)

plt.figure(figsize=(11,9))
plt.axis ([-10, 10, -10, 10])
plt.figure(2)
plt.scatter(o1,o2)
plt.grid()



def AWGN(SNR, A, Tb, num = bits):
  N0 =  A**2 * Tb / 10**(SNR / 10)
  X = np.random.normal(0, np.sqrt(N0 / 2), size = num)
  Y = np.random.normal(0, np.sqrt(N0 / 2), size = num)
  #AWGN
  Z = X + 1j*Y 
  return Z
  
# Στοχαστικές διαδικασίες θορύβου 
Z1 = AWGN(5, Amplitude, Tb)
Z2 = AWGN(15, Amplitude, Tb)

# Received signals
r1 = BPAM + Z1
r2 = BPAM + Z2

# Γραφικές με το Re{Noise}
plt.figure(figsize=(11,9))
plt.figure(3);
plt.step(t,np.real(r1));

plt.figure(figsize=(11,9))
plt.figure(4);
plt.step(t,np.real(r2));

plt.figure(figsize=(11,9))
plt.figure(10);
plt.plot(t,np.real(r1));

plt.figure(figsize=(11,9))
plt.figure(5);
plt.grid();
plt.scatter(np.real(r1), np.imag(r1), label='Received');
plt.scatter(np.real(BPAM), np.imag(BPAM), color='r', label='Transmitted');
plt.annotate('1', xy=(-Amplitude, 0), xytext=(-Amplitude + 0.1, 0));
plt.annotate('0', xy=(Amplitude, 0), xytext=(Amplitude + 0.1, 0));
plt.xlim([-Amplitude - 1, Amplitude + 1]);
plt.ylim([-Amplitude - 1, Amplitude + 1]);
plt.legend();

plt.figure(figsize=(11,9))
plt.figure(6);
plt.grid();
plt.scatter(np.real(r2), np.imag(r2), label='Received');
plt.scatter(np.real(BPAM), np.imag(BPAM), color='r', label='Transmitted');
plt.annotate('1', xy=(-Amplitude, 0), xytext=(-Amplitude + 0.1, 0));
plt.annotate('0', xy=(Amplitude, 0), xytext=(Amplitude + 0.1, 0));
plt.xlim([-Amplitude - 1, Amplitude + 1]);
plt.ylim([-Amplitude - 1, Amplitude + 1]);
plt.legend();


def BER(samplesNum = 10**5, log_log=False):
  m1 = np.random.randint(2, size = samplesNum) 
  polar = lambda x : -Amplitude if x == 0 else Amplitude
  s = [polar(x) for x in m1] 
  decode = lambda x: 1 if (x.real > 0) else 0
  SNR = np.arange(0,16,1)
  BER = np.zeros(16) 
  for i in SNR:
    Z = AWGN(i, Amplitude, 1, samplesNum)
    r = Z + s
    decoded = np.array([decode(t) for t in r])
    
    numOfErrors = (m1 ^ decoded)
    totalErrors = np.sum(numOfErrors)
    BER[i] = totalErrors*1.0/samplesNum
  plt.figure();
  plt.plot(SNR, 10*np.log10(BER) if log_log else BER, lw=0, marker='o', 
          label='piramatikoBER');
  
  #υπολογισμός θεωρητικού BER
  BER_th = 0.5*erfc(np.sqrt(10**(SNR/10)))
  plt.plot(SNR, 10*np.log10(BER_th) if log_log else BER_th, 
           label='theoritiko BER');
  plt.legend();
  
  plt.xlabel('$E_b / N_0$ (dB)');
  plt.ylabel('BER');
  
  return SNR, BER, BER_th

BER();


# In[3]:


##################################################################################################################### ερώτημα 4
SNR1 = 5  
SNR2 = 15
samples=1
if 2 % 2 == 0: 
    fc=2 
else: 
    fc=3
#####################################################################################################################
#####################################################################################################################
#πάρακατω παίρνουμε το QPSK modulation αυτόυσιο απο το προηγόυμενο ερώτημα απλά με μειωμένα samples
#####################################################################################################################
#sequence2=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r']
sequence2= [None] * 26
for i in range (0, bits,2):                          
    if sequence[i]==0 and sequence[i+1]==0  :  
        sequence2[i//2]='00'
    elif sequence[i]==0 and sequence[i+1]==1  :
        sequence2[i//2]='01'
    elif sequence[i]==1 and sequence[i+1]==1  :
        sequence2[i//2]='11'
    else: sequence2[i//2]='10'

signal1=np.zeros(samples*bits)
signalRe=np.zeros(samples*bits) 
signalIm=np.zeros(samples*bits)
signalRe2=np.zeros(samples*bits)
signalIm2=np.zeros(samples*bits)
time= np.linspace(0,1/fm,46)

#Θέτουμε την ΄φαση π/4,3π/4,5π/4,7π/4 ανάλογα με την τιμή κάθε ζεύγους ψηφίων της ακολουθίας
#και ορίζουμε την αντίστοιχη τιμή του σήματος
for i in range (0, samples*bits,2):                                  
    if sequence[i//1]==0 and sequence[(i+1)//1]==0  : 
        phase=np.pi/4                                          
    elif sequence[i//1]==0 and sequence[(i+1)//1]==1  : 
        phase=np.pi*3/4
    elif sequence[i//1]==1 and sequence[(i+1)//1]==1  :
        phase=np.pi*5/4
    else: phase=np.pi*7/4
    for j in range (i,i+2,1):
        signal1[j]=Amplitude*np.cos(2*np.pi*fc*time[j]+phase)
        signalRe[j]=Amplitude*np.cos(phase)              
        signalIm[j]=Amplitude*np.sin(phase)
        signalRe2[j]=Amplitude*np.cos(phase-np.pi/4)
        signalIm2[j]=Amplitude*np.sin(phase-np.pi/4)
#####################################################################################################################
#Δημιουργούμε το διάγραμμα αστερισμού του QPSK
plt.plot (signalRe2,signalIm2,'bo' , label = 'σύμβ. χωρίς κωδ. π/4')   
plt.plot (signalRe,signalIm,'rx' , label = 'σύμβ. με κωδ. π/4')     
plt.axis ([-Amplitude*2,Amplitude*2,-Amplitude-1,Amplitude+1])                           
plt.xlabel("real")                                    
plt.ylabel("imaginary")                     
plt.title("QPSK-π/4 Gray")
plt.annotate('11', xy=(-Amplitude, 0), xytext=(-Amplitude + 0.1, 0));
plt.annotate('00', xy=(Amplitude, 0), xytext=(Amplitude + 0.1, 0));
plt.annotate('10', xy=(0, -Amplitude), xytext=(0, -Amplitude + 0.1));
plt.annotate('01', xy=(0, Amplitude), xytext=(0, Amplitude + 0.1));
plt.annotate('00', xy=(Amplitude/np.sqrt(2), Amplitude/np.sqrt(2)), xytext=(Amplitude/np.sqrt(2), Amplitude/np.sqrt(2) + 0.1));
plt.annotate('11', xy=(-Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2)), xytext=(-Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2) + 0.1));
plt.annotate('01', xy=(-Amplitude/np.sqrt(2), Amplitude/np.sqrt(2)), xytext=(-Amplitude/np.sqrt(2), Amplitude/np.sqrt(2) + 0.1));
plt.annotate('10', xy=(Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2)), xytext=(Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2) + 0.1));
plt.legend() 
plt.grid()
#plt.savefig('export_images/3_constellationdiagram.png') 
plt.show()      

#################################################################################################################### 3β ερώτημα
#Δημιουργούμε θόρυβο Eb/N0 = 5db
powerNoise= Amplitude**2/(2*math.sqrt(10.0**(SNR1/10.0)))  
#Τον προσθέτουμε στα ανάλογα σήματα
noise = np.random.normal(0,math.sqrt(powerNoise) ,samples*bits)  
signalNoiseRe2=noise+signalRe2
signalNoiseRe=noise+signalRe
noiseRe=np.random.normal(0,math.sqrt(powerNoise) ,samples*bits)+signalIm
noiseRe2=np.random.normal(0,math.sqrt(powerNoise) ,samples*bits)+signalIm2

#Διάγραμμα αστερισμού του QPSK π/4 με SNR 5db
plt.plot(signalRe,signalIm,'rx',label = 'Transmitted')            
plt.plot (signalNoiseRe,noiseRe,'gx' , label = 'Received')     
plt.grid()                                                  
plt.xlabel("Πραγματικό Μέρος")                                        
plt.ylabel("Φανταστικό μέρος")            
plt.title("Διάγραμμα Αστερισμού QPSK-π/4 με AWGN θόρυβο των 5db SNR")
plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
#plt.savefig('export_images/3_β1.png') 
plt.show()
#Δημιουργούμε θόρυβο Eb/N0 = 15db
powerNoise= Amplitude**2/(2*math.sqrt(10.0**(SNR2/10.0)))
#Τον προσθέτουμε στα ανάλογα σήματα
noise = np.random.normal(0,math.sqrt(powerNoise) ,samples*bits)
signalNoiseRe2=noise+signalRe2
signalNoiseRe=noise+signalRe
noiseRe=np.random.normal(0,math.sqrt(powerNoise) ,samples*bits)+signalIm
noiseRe2=np.random.normal(0,math.sqrt(powerNoise) ,samples*bits)+signalIm2

#Διάγραμμα αστερισμού του QPSK π/4 με SNR 15db
plt.plot(signalRe,signalIm,'rx',label = 'Transmitted')                                                          
plt.plot (signalNoiseRe,noiseRe,'gx' , label = 'Received')       
plt.grid()                                                  
plt.xlabel("Πραγματικό Μέρος")                                        
plt.ylabel("Φανταστικό μέρος")             
plt.title("Διάγραμμα Αστερισμού QPSK-π/4 με AWGN θόρυβο των 15db SNR")
plt.legend()
#plt.savefig('export_images/3_β2.png') 
plt.show()

#####################################################################################################################ερώτημα 3γ
samples=50
#####################################################################################################################
#####################################################################################################################
#πάρακατω παίρνουμε το QPSK modulation αυτόυσιο απο το προηγόυμενο ερώτημα απλά με μειωμένα samples
#####################################################################################################################
#sequence2=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r']
sequence2=[None]*26
for i in range (0, bits,2):                          
    if sequence[i]==0 and sequence[i+1]==0  :  
        sequence2[i//2]='00'
    elif sequence[i]==0 and sequence[i+1]==1  :
        sequence2[i//2]='01'
    elif sequence[i]==1 and sequence[i+1]==1  :
        sequence2[i//2]='11'
    else: sequence2[i//2]='10'
    
signal1=np.zeros(samples*bits)#afta einai gia to 3
signalRe=np.zeros(samples*bits) 
signalIm=np.zeros(samples*bits)
signalRe2=np.zeros(samples*bits)
signalIm2=np.zeros(samples*bits)
time= np.linspace(0,1/fm,2300)

for i in range (0, samples*bits,100):                                    
    if sequence[i//50]==0 and sequence[(i+50)//50]==0  :
        phase=np.pi/4                                          
    elif sequence[i//50]==0 and sequence[(i+50)//50]==1  : 
        phase=np.pi*3/4
    elif sequence[i//50]==1 and sequence[(i+50)//50]==1  :
        phase=np.pi*5/4
    else: phase=np.pi*7/4
    for j in range (i,i+100,1):
        signal1[j]=Amplitude*np.cos(2*np.pi*fc*time[j]+phase)
        signalRe[j]=Amplitude*np.cos(phase)              
        signalIm[j]=Amplitude*np.sin(phase)
        signalRe2[j]=Amplitude*np.cos(phase-np.pi/4)
        signalIm2[j]=Amplitude*np.sin(phase-np.pi/4)
#####################################################################################################################
counter = np.zeros(bits)
for i in range (0, bits):
    expNoise = (Amplitude**2/(10**(i/10.0)))/2.0
    expNoiseRe = np.random.normal(0, math.sqrt(expNoise), 50*bits)
    expNoiseIm = np.random.normal(0, math.sqrt(expNoise), 50*bits)
    expQpskRe = signalRe2 + expNoiseRe
    expQpskIm = signalIm2 + expNoiseIm
    expQpskRe2 = signalRe + expNoiseRe
    expQpskIm2 = signalIm + expNoiseIm
    #εάν η απόσταση μεταξύ του σημείου που μεταδίδουμε και της πραγματικής τιμής είναι μεγαλύτερη απο την τιμή sqrt(2)*Amplitude/2 έχουμε σφάλμα
    for j in range (0, samples*bits, 2):           
        if (math.sqrt((expQpskRe[j]-signalRe2[j])**2 + (expQpskIm[j]-signalIm2[j])**2) > Amplitude/2.0*math.sqrt(2)):
            counter[i]+=1
        if (math.sqrt((expQpskRe2[j+1]-signalRe[j+1])**2 + (expQpskIm2[j+1]-signalIm[j+1])**2) > Amplitude/2.0*math.sqrt(2)):
            counter[i]+=1
            



# In[90]:


#################################################
#4g)
from scipy import *
from math import sqrt, ceil  # scalar calls are faster
from scipy.special import erfc
import matplotlib.pyplot as plt

rand   = random.rand
normal = random.normal

SNR_MIN   = 0
SNR_MAX   = 10
FrameSize = 46
Eb_No_dB  = arange(SNR_MIN,SNR_MAX+1)
Eb_No_lin = 10**(Eb_No_dB/10.0)  # linear SNR

# Allocate memory
Pe        = empty(shape(Eb_No_lin))
BER       = empty(shape(Eb_No_lin))

# signal vector (for faster exec we can repeat the same frame)
s = 2*random.randint(0,high=2,size=FrameSize)-1
im = 2*random.randint(0,high=2,size=FrameSize)-1
loop = 0
for snr in Eb_No_lin:
     No        = 1.0/snr
     Pe[loop]  = 0.5*erfc(sqrt(snr))
     nFrames   = ceil(100.0/FrameSize/Pe[loop])
     error_sum = 0
     error_sum1=0
     scale = sqrt(No/2)

     for frame in arange(nFrames):
   # noise
       n = normal(scale=scale, size=FrameSize)

   # received signal + noise
       x = s + n
       xim=im+n
   # detection (information is encoded in signal phase)
       y = sign(x)
       y1 = sign(xim)
   # error counting
       err = where (y != s)
       err1 = where (y != s)
       error_sum += len(err[0])
       error_sum1 += len(err[0])
       final=(error_sum1+error_sum)/2

   # end of frame loop
   ##################################################

     BER[loop] = error_sum/(FrameSize*nFrames)  # SNR loop level
     #print 'Eb_No_dB=%2d, BER=%10.4e, Pe[loop]=%10.4e' % \
     print(Eb_No_dB[loop], BER[loop], Pe[loop])
     loop += 1

plt.semilogy(Eb_No_dB, Pe,'r',linewidth=2)
plt.semilogy(Eb_No_dB, BER,'-s')
plt.grid(True)
plt.legend(('analytical','simulation'))
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.show()


# In[193]:


import binascii
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

f = open(r"C:\Users\ellin\Desktop\Telecommunications\shannon_odd.txt", "r")

line = f.read()

f.close()

sequence=bin(int.from_bytes(line.encode(), 'big'))

def string2bits(s=''):
    return [bin(ord(x))[2:].zfill(8) for x in s]

def bits2string(b=None):
    return ''.join([chr(int(x, 2)) for x in b])

def BinaryToDecimal(binary): 
    value = 0

    for i in range(len(b_num)):
        digit = b_num.pop()
        if digit == '1':
            value = value + pow(2, i)
        print("The decimal value of the number is", value)

b = string2bits(line)
c=list()
for i in range(0, len(b)): 
    #print(b[i])
    b[i] = int(b[i], 2) 
    #print(binaryToDecimal(int(b[i])))
    #print(b[i])
time = np.linspace(0,1,len(b)) # δημ. του άξονα χρόνου για την δειγματοληψία
a = np.array(b)
#ορίσματα της εξίσωσης mid riser κβαντιστή
bits=8
Levels = 2**bits    
d=1/(Levels) # βήματα κβάντισης
signal=d*np.floor(a/d) + d/2


#Κώδικας gray
def gray_code(n):
    if n <= 0:
        return []
    if n == 1:
        return ['0', '1']
    res = gray_code(n-1)
    return ['0'+s for s in res] + ['1'+s for s in res[::-1]]
#Κάλεσμα συνάρτησης gray με το num
g = gray_code(bits)
plt.figure(figsize=(12,9)) # κβαντισμένο δειγματοληπτημένο σήμα με fs1 
#plt.figure(6)
#plt.grid()
#plt.yticks(np.arange(0.0,4.0+d/2,d),g[0:Levels]) 
plt.stem(time,signal,':',basefmt=" ",use_line_collection=True);
#plt.plot(time,signal)
#plt.show


# In[45]:


s=list(sequence)
s[1]='0'
"".join(s)
sequence="".join(s)
sequence


# In[194]:


Amplitude=1
fm=4000
signal1=np.zeros(len(sequence)+3)
signalRe=np.zeros(len(sequence)+3) 
signalIm=np.zeros(len(sequence)+3)
signalRe2=np.zeros(len(sequence)+3)
signalIm2=np.zeros(len(sequence)+3)
time= np.linspace(0,1/fm,len(sequence)+3)

for i in range (0, len(sequence),2): 
    if sequence[i]=='0' and sequence[(i+1)]=='0'  : 
        phase=np.pi/4                                          
    elif sequence[i]=='0' and sequence[(i+1)]=='1'  : 
        phase=np.pi*3/4
    elif sequence[i//1]=='1' and sequence[(i+1)//1]=='1'  :
        phase=np.pi*5/4
    else: phase=np.pi*7/4
    a=i
    for j in range (a,a+2,1):
        signal1[j]=Amplitude*np.cos(2*np.pi*time[j]+phase)
        signalRe[j]=Amplitude*np.cos(phase)              
        signalIm[j]=Amplitude*np.sin(phase)
    if i==5094:
        break
plt.plot (signalRe[signalRe!=0],signalIm[signalIm!=0],'bo')   
plt.xlabel("real")                                    
plt.ylabel("imaginary")                     
plt.title("QPSK Gray")
plt.annotate('11', xy=(-Amplitude, 0), xytext=(-Amplitude + 0.1, 0));
plt.annotate('00', xy=(Amplitude, 0), xytext=(Amplitude + 0.1, 0));
plt.annotate('10', xy=(0, -Amplitude), xytext=(0, -Amplitude + 0.1));
plt.annotate('01', xy=(0, Amplitude), xytext=(0, Amplitude + 0.1));
plt.annotate('00', xy=(Amplitude/np.sqrt(2), Amplitude/np.sqrt(2)), xytext=(Amplitude/np.sqrt(2), Amplitude/np.sqrt(2) + 0.1));
plt.annotate('11', xy=(-Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2)), xytext=(-Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2) + 0.1));
plt.annotate('01', xy=(-Amplitude/np.sqrt(2), Amplitude/np.sqrt(2)), xytext=(-Amplitude/np.sqrt(2), Amplitude/np.sqrt(2) + 0.1));
plt.annotate('10', xy=(Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2)), xytext=(Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2) + 0.1));
plt.legend() 
plt.grid()
#plt.savefig('export_images/3_constellationdiagram.png') 
plt.show()      


# In[55]:


#Δημιουργούμε θόρυβο Eb/N0 = 5db
SNR1=5
powerNoise= Amplitude**2/(2*math.sqrt(10.0**(SNR1/10.0)))  
#Τον προσθέτουμε στα ανάλογα σήματα
noise = np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)  
#signalNoiseRe2=noise+signalRe2
signalNoiseRe=noise+signalRe
noiseRe=np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)+signalIm
#noiseRe2=np.random.normal(0,math.sqrt(powerNoise) ,len(sequence))+signalIm2

#Διάγραμμα αστερισμού του QPSK π/4 με SNR 5db
            
plt.plot (signalNoiseRe[signalNoiseRe != 0],noiseRe[noiseRe != 0],'gx' , label = 'Received')     
plt.plot(signalRe[signalRe != 0],signalIm[signalIm!=0],'rx',label = 'Transmitted')
plt.grid()                                                  
plt.xlabel("Πραγματικό Μέρος")                                        
plt.ylabel("Φανταστικό μέρος")            
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN θόρυβο των 5db SNR")
plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
#plt.savefig('export_images/3_β1.png') 
plt.show()


# In[56]:


signalNoiseRe.shape


# In[58]:


#telefteo erotima
import numpy as np
extractedseq=np.array([])
x2=signalNoiseRe
y2=noiseRe
for i in range(0,len(signalNoiseRe),2):
    d0 = np.sqrt(np.power(x2[i] - (-np.sqrt(2) / 2), 2) + np.power(y2[i] - (-np.sqrt(2) / 2), 2));
    d1 = np.sqrt(np.power(x2[i] - (-np.sqrt(2) / 2), 2) + np.power(y2[i] -   (np.sqrt(2) / 2),  2));
    d2 = np.sqrt(np.power(x2[i] - ( np.sqrt(2) / 2), 2) + np.power(y2[i] - ( np.sqrt(2) / 2), 2));
    d3 = np.sqrt(np.power(x2[i] -   (np.sqrt(2) / 2) , 2) + np.power(y2[i] - (-np.sqrt(2) / 2), 2));
    small=10000;
    symbol=0;
    a0=(signalNoiseRe[i] - d0)
    if a0<=small:
        small=signalNoiseRe[i]-d0
        symbol=0
    if signalNoiseRe[i]-d1<small:
        small=signalNoiseRe[i]-d1
        symbol=1
    if signalNoiseRe[i]-d2<small:
        small=signalNoiseRe[i]-d2
        symbol=2
    if signalNoiseRe[i]-d3<small:
        small=signalNoiseRe[i]-d3
        symbol=3
    
    if symbol==0:
        extractedseq=np.append(extractedseq,[0,0])
    elif symbol==1:
        extractedseq=np.append(extractedseq,[0,1]) 
    elif symbol==2:
        extractedseq=np.append(extractedseq,[1,0])
    elif symbol==3:
        extractedseq=np.append(extractedseq,[1,1])
    #print(extractedseq.shape)   
extractedseq.astype(int)
len(sequence)

string=''
a=''
for i in range(0,len(extractedseq)):
    #for(j) in range(i,i+8):
    a=a+str(int(extractedseq[i]))
    #print(a)
    #ch=bits2string(int(a))
string=int(a).to_bytes((int(a).bit_length() + 7) // 8, 'big').decode()
print(string)
#string=string+ch
print(string)
string
a


# In[195]:


#Δημιουργούμε θόρυβο Eb/N0 = 15db
SNR2=15
powerNoise= Amplitude**2/(2*math.sqrt(10.0**(SNR2/10.0)))  
#Τον προσθέτουμε στα ανάλογα σήματα
noise = np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)  
signalNoiseRe=noise+signalRe
noiseRe=np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)+signalIm

#Διάγραμμα αστερισμού του QPSK με SNR 15db
            
plt.plot (signalNoiseRe[signalNoiseRe != 0],noiseRe[noiseRe != 0],'gx' , label = 'Received')     
plt.plot(signalRe[signalRe != 0],signalIm[signalIm!=0],'rx',label = 'Transmitted')
plt.grid()                                                  
plt.xlabel("Πραγματικό Μέρος")                                        
plt.ylabel("Φανταστικό μέρος")            
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN θόρυβο των 15db SNR")
plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
#plt.savefig('export_images/3_β1.png') 
plt.show()


# In[81]:


from scipy import *
from math import sqrt, ceil  # scalar calls are faster
from scipy.special import erfc
import matplotlib.pyplot as plt

rand   = random.rand
normal = random.normal

SNR_MIN   = 0
SNR_MAX   = 10
FrameSize = 5100
Eb_No_dB  = arange(SNR_MIN,SNR_MAX+1)
Eb_No_lin = 10**(Eb_No_dB/10.0)  # linear SNR

# Allocate memory
Pe        = empty(shape(Eb_No_lin))
BER       = empty(shape(Eb_No_lin))

# signal vector (for faster exec we can repeat the same frame)
s = 2*random.randint(0,high=2,size=FrameSize)-1
im=2*random.randint(0,high=2,size=FrameSize)-1
loop = 0
for snr in Eb_No_lin:
     No        = 1.0/snr
     Pe[loop]  = 0.5*erfc(sqrt(snr))
     nFrames   = ceil(100.0/FrameSize/Pe[loop])
     error_sum = 0
     error_sum1=0
     scale = sqrt(No/2)

     for frame in arange(nFrames):
   # noise
       n = normal(scale=scale, size=FrameSize)

   # received signal + noise
       x = s + n
       xim=im+nhttp://localhost:8888/notebooks/Untitled23.ipynb#
   # detection (information is encoded in signal phase)
       y = sign(x)
       y1 = sign(xim)
   # error counting
       err = where (y != s)
       err1 = where (y != s)
       error_sum += len(err[0])
       error_sum1 += len(err[0])
       final=(error_sum1+error_sum)/2

   # end of frame loop
   ##################################################

     BER[loop] = final/(FrameSize*nFrames)  # SNR loop level
     #print 'Eb_No_dB=%2d, BER=%10.4e, Pe[loop]=%10.4e' % \
     print(Eb_No_dB[loop], BER[loop], Pe[loop])
     loop += 1

plt.semilogy(Eb_No_dB, Pe,'r',linewidth=2)
plt.semilogy(Eb_No_dB, BER,'-s')
plt.grid(True)
plt.legend(('analytical','simulation'))
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.show()


# In[85]:


from scipy import *
from math import sqrt, ceil  # scalar calls are faster
from scipy.special import erfc
import matplotlib.pyplot as plt

rand   = random.rand
normal = random.normal

SNR_MIN   = 0
SNR_MAX   = 10
FrameSize = 5100
Eb_No_dB  = arange(SNR_MIN,SNR_MAX+1)
Eb_No_lin = 10**(Eb_No_dB/10.0)  # linear SNR

# Allocate memory
Pe        = empty(shape(Eb_No_lin))
BER       = empty(shape(Eb_No_lin))

# signal vector (for faster exec we can repeat the same frame)
s = 2*random.randint(0,high=2,size=FrameSize)-1
im=2*random.randint(0,high=2,size=FrameSize)-1
loop = 0
for snr in Eb_No_lin:
     No        = 1.0/snr
     Pe[loop]  = 0.5*erfc(sqrt(snr))
     nFrames   = ceil(100.0/FrameSize/Pe[loop])
     error_sum = 0
     error_sum1=0
     scale = sqrt(No/2)

     for frame in arange(nFrames):
   # noise
       n = normal(scale=scale, size=FrameSize)

   # received signal + noise
       x = s + n
       xim=im+n
   # detection (information is encoded in signal phase)
       y = sign(x)
       y1 = sign(xim)
   # error counting
       err = where (y != s)
       err1 = where (y != s)
       error_sum += len(err[0])
       error_sum1 += len(err[0])
       final=(error_sum1+error_sum)/2

   # end of frame loop
   ##################################################

     BER[loop] = error_sum/(FrameSize*nFrames)  # SNR loop level
     #print 'Eb_No_dB=%2d, BER=%10.4e, Pe[loop]=%10.4e' % \
     print(Eb_No_dB[loop], BER[loop], Pe[loop])
     loop += 1

plt.semilogy(Eb_No_dB, Pe,'r',linewidth=2)
plt.semilogy(Eb_No_dB, BER,'-s')
plt.grid(True)
plt.legend(('analytical','simulation'))
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.show()


# In[16]:


print(signal1)


# In[236]:


from scipy.io import wavfile
samplerate, data = wavfile.read(r"C:\Users\ellin\Desktop\Telecommunications\soundfile1_lab2.wav")


# In[238]:


t1 = np.linspace(0, 1/samplerate, len(data))

plt.figure(figsize=(12,9))
plt.figure(2)
#plt.stem(t1,signal,':',basefmt=" ",use_line_collection=True);
plt.plot(t1,data);


# In[239]:


bits=8       

#ορίσματα της εξίσωσης mid riser κβαντιστή

Levels = 2**bits    
d=(max(data)-min(data))/(Levels) # βήματα κβάντισης
signal=d*np.floor(data/d) + d/2
#εξίσωση mid-riser κβαντιστή, χρήση της floor ώστε κατα τον κβαντισμό τα 
# σημεία της συνάρτησης να φτάνουν στο πλησιέστερο επίπεδο.
#==============================================================================
# quant_signal=d*np.floor(fs1_y/d) + d/2
#==============================================================================

#Κώδικας gray
def gray_code(n):
    if n <= 0:
        return []
    if n == 1:
        return ['0', '1']
    res = gray_code(n-1)
    return ['0'+s for s in res] + ['1'+s for s in res[::-1]]
#Κάλεσμα συνάρτησης gray με το num
g = gray_code(bits)

plt.figure(figsize=(25,25)) # κβαντισμένο δειγματοληπτημένο σήμα με fs1 
#plt.figure(6)
plt.grid()
plt.yticks(np.arange(0.0,max(signal)+d/2,d),g[0:Levels]) 
plt.stem(t1,signal,':',basefmt=" ",use_line_collection=True);
#plt.plot(t1,data);


# In[240]:


sequence=[]
a=np.arange(0.0,max(signal),d)
r=list(a)
#print(a)
#x = np.where(a == signal[0])
#x
for x in signal:
    for j in range(len(r)):
        if((x+d/2)==r[j]):
            sequence += '{:08b}'.format(j)
            print(j)


# In[242]:


Amplitude=1
fm=4000
signal1=np.zeros(len(sequence)+3)
signalRe=np.zeros(len(sequence)+3) 
signalIm=np.zeros(len(sequence)+3)
signalRe2=np.zeros(len(sequence)+3)
signalIm2=np.zeros(len(sequence)+3)
time= np.linspace(0,1/fm,len(sequence)+3)

for i in range (0, len(sequence),2): 
    if sequence[i]=='0' and sequence[(i+1)]=='0'  : 
        phase=np.pi/4                                          
    elif sequence[i]=='0' and sequence[(i+1)]=='1'  : 
        phase=np.pi*3/4
    elif sequence[i//1]=='1' and sequence[(i+1)//1]=='1'  :
        phase=np.pi*5/4
    else: phase=np.pi*7/4
    a=i
    for j in range (a,a+2,1):
        signal1[j]=Amplitude*np.cos(2*np.pi*time[j]+phase)
        signalRe[j]=Amplitude*np.cos(phase)              
        signalIm[j]=Amplitude*np.sin(phase)
    if i==329411:
        break
plt.plot (signalRe[signalRe!=0],signalIm[signalIm!=0],'bo')   
plt.xlabel("real")                                    
plt.ylabel("imaginary")                     
plt.title("QPSK Gray")
plt.annotate('00', xy=(Amplitude/np.sqrt(2), Amplitude/np.sqrt(2)), xytext=(Amplitude/np.sqrt(2), Amplitude/np.sqrt(2) + 0.1));
plt.annotate('11', xy=(-Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2)), xytext=(-Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2) + 0.1));
plt.annotate('01', xy=(-Amplitude/np.sqrt(2), Amplitude/np.sqrt(2)), xytext=(-Amplitude/np.sqrt(2), Amplitude/np.sqrt(2) + 0.1));
plt.annotate('10', xy=(Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2)), xytext=(Amplitude/np.sqrt(2), -Amplitude/np.sqrt(2) + 0.1));
#plt.legend() 
plt.grid()
#plt.savefig('export_images/3_constellationdiagram.png') 
plt.show()      


# In[204]:


#Δημιουργούμε θόρυβο Eb/N0 = 15db
SNR2=14
powerNoise= Amplitude**2/(2*math.sqrt(10.0**(SNR2/10.0)))  
#Τον προσθέτουμε στα ανάλογα σήματα
noise = np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)  
signalNoiseRe=noise+signalRe
noiseRe=np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)+signalIm

#Διάγραμμα αστερισμού του QPSK με SNR 15db
            
plt.plot (signalNoiseRe[signalNoiseRe != 0],noiseRe[noiseRe != 0],'gx' , label = 'Received')     
plt.plot(signalRe[signalRe != 0],signalIm[signalIm!=0],'rx',label = 'Transmitted')
plt.grid()                                                  
plt.xlabel("Πραγματικό Μέρος")                                        
plt.ylabel("Φανταστικό μέρος")            
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN θόρυβο των 14db SNR")
plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
#plt.savefig('export_images/3_β1.png') 
plt.show()


# In[244]:


#Δημιουργούμε θόρυβο Eb/N0 = 4db
SNR2=4
powerNoise= Amplitude**2/(2*math.sqrt(10.0**(SNR2/10.0)))  
#Τον προσθέτουμε στα ανάλογα σήματα
noise = np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)  
signalNoiseRe1=noise+signalRe
noiseRe1=np.random.normal(0,math.sqrt(powerNoise) ,len(sequence)+3)+signalIm

#Διάγραμμα αστερισμού του QPSK με SNR 15db
            
plt.plot (signalNoiseRe1[signalNoiseRe1 != 0],noiseRe1[noiseRe1 != 0],'gx' , label = 'Received')     
plt.plot(signalRe[signalRe != 0],signalIm[signalIm!=0],'rx',label = 'Transmitted')
plt.grid()                                                  
plt.xlabel("Πραγματικό Μέρος")                                        
plt.ylabel("Φανταστικό μέρος")            
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN θόρυβο των 4db SNR")
plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
#plt.savefig('export_images/3_β1.png') 
plt.show()


# In[232]:


s = 2*random.randint(0,high=2,size=1360963)-1
im=2*random.randint(0,high=2,size=1360963)-1
snr=4
error_sum1=0
No=1/snr
scale = sqrt(No/2)
n = normal(scale=scale, size=1360963)
Pe  = 0.5*erfc(sqrt(snr))
nFrames   = ceil(100.0/1360963/Pe)
error_sum = 0
error_sum1=0
   # received signal + noise
x = s + n
xim=im+n

y = np.sign(x)
y1 = np.sign(xim)

err = where (y != s)
err1 = where (y1 != s)

error_sum += len(err[0])
error_sum1 += len(err1[0])

final=(error_sum1+error_sum)/2

BER = final/(FrameSize*nFrames)

print(BER,Pe)


# In[233]:


s = 2*random.randint(0,high=2,size=1360963)-1
im=2*random.randint(0,high=2,size=1360963)-1
snr=14
error_sum1=0
No=1/snr
scale = sqrt(No/2)
n = normal(scale=scale, size=1360963)
Pe  = 0.5*erfc(sqrt(snr))
nFrames   = ceil(100.0/1360963/Pe)
error_sum = 0
error_sum1=0
   # received signal + noise
x = s + n
xim=im+n

y = np.sign(x)
y1 = np.sign(xim)

err = where (y != s)
err1 = where (y1 != s)

error_sum += len(err[0])
error_sum1 += len(err1[0])

final=(error_sum1+error_sum)/2

BER = final/(FrameSize*nFrames)

print(BER,Pe)


# In[258]:


len(signalNoiseRe1)


# In[252]:


import numpy as np
extractedseq=np.array([])
x2=signalNoiseRe
y2=noiseRe
for i in range(0,len(signalNoiseRe),2):
    d0 = np.sqrt(np.power(x2[i] - (-np.sqrt(2) / 2), 2) + np.power(y2[i] - (-np.sqrt(2) / 2), 2));
    d1 = np.sqrt(np.power(x2[i] - (-np.sqrt(2) / 2), 2) + np.power(y2[i] -   (np.sqrt(2) / 2),  2));
    d2 = np.sqrt(np.power(x2[i] - ( np.sqrt(2) / 2), 2) + np.power(y2[i] - ( np.sqrt(2) / 2), 2));
    d3 = np.sqrt(np.power(x2[i] -   (np.sqrt(2) / 2) , 2) + np.power(y2[i] - (-np.sqrt(2) / 2), 2));
    small=10000;
    symbol=0;
    a0=(signalNoiseRe[i] - d0)
    if a0<=small:
        small=signalNoiseRe[i]-d0
        symbol=0
    if signalNoiseRe[i]-d1<small:
        small=signalNoiseRe[i]-d1
        symbol=1
    if signalNoiseRe[i]-d2<small:
        small=signalNoiseRe[i]-d2
        symbol=2
    if signalNoiseRe[i]-d3<small:
        small=signalNoiseRe[i]-d3
        symbol=3
    
    if symbol==0:
        extractedseq=np.append(extractedseq,[0,0])
    elif symbol==1:
        extractedseq=np.append(extractedseq,[0,1]) 
    elif symbol==2:
        extractedseq=np.append(extractedseq,[1,0])
    elif symbol==3:
        extractedseq=np.append(extractedseq,[1,1])
    print(i,symbol)
    #print(extractedseq.shape)   
extractedseq.astype(int)
len(sequence)


# In[ ]:


import numpy as np
extractedseq=np.array([])
x2=signalNoiseRe1
y2=noiseRe1
for i in range(0,len(signalNoiseRe),2):
    d0 = np.sqrt(np.power(x2[i] - (-np.sqrt(2) / 2), 2) + np.power(y2[i] - (-np.sqrt(2) / 2), 2));
    d1 = np.sqrt(np.power(x2[i] - (-np.sqrt(2) / 2), 2) + np.power(y2[i] -   (np.sqrt(2) / 2),  2));
    d2 = np.sqrt(np.power(x2[i] - ( np.sqrt(2) / 2), 2) + np.power(y2[i] - ( np.sqrt(2) / 2), 2));
    d3 = np.sqrt(np.power(x2[i] -   (np.sqrt(2) / 2) , 2) + np.power(y2[i] - (-np.sqrt(2) / 2), 2));
    small=10000;
    symbol=0;
    a0=(signalNoiseRe[i] - d0)
    if a0<=small:
        small=signalNoiseRe[i]-d0
        symbol=0
    if signalNoiseRe[i]-d1<small:
        small=signalNoiseRe[i]-d1
        symbol=1
    if signalNoiseRe[i]-d2<small:
        small=signalNoiseRe[i]-d2
        symbol=2
    if signalNoiseRe[i]-d3<small:
        small=signalNoiseRe[i]-d3
        symbol=3
    
    if symbol==0:
        extractedseq=np.append(extractedseq,[0,0])
    elif symbol==1:
        extractedseq=np.append(extractedseq,[0,1]) 
    elif symbol==2:
        extractedseq=np.append(extractedseq,[1,0])
    elif symbol==3:
        extractedseq=np.append(extractedseq,[1,1])
    print(i,symbol)
    #print(extractedseq.shape)   
extractedseq.astype(int)
len(sequence)


# In[ ]:




