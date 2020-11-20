import numpy as np
import pandas as pd
def dynamicpricingbill():
	arr=[31,28,31,30,31,30,31,31,30,31,30,31]
	arr1=['January','Februray','March','April','May','June','July','August','September','October','November','December']
	billamount=0.0
	flag=0
	df=pd.read_csv('sample1.csv')
	df=np.array(df)
	x=df.shape[0]-169
	ticks=0
	p1=0
	Cc=float(input('Enter price of coal per unit'))
	Cd=float(input('Enter price of diesel per unit'))
	Cg=float(input('Enter price of gas per unit'))	
	Pc=float(input('Enter percentage of coal usedused'))
	Pd=float(input('Enter percentage of diesel used'))
	Pg=float(input('Enter percentage of gas used'))
	fp=float(input('Enter fixed (infrastructure) charges:-'))
	price_one_unit_electricity=Cc*Pc+Cd*Pd+Cg*Pg+fp
	print()
	data=[]
	for i in range(0,x):
		consumption=df[i:168+i,2]
		consumption = consumption.astype(np.float)
		avgcon=np.sum(consumption)
		# avgcon=float(avgcon)
		avgcon/=float(168)
		D=float(df[169+i][2])
		D/=avgcon
		D*=price_one_unit_electricity
		if flag ==0:
			totalwattsused=df[0:168,1]
			totalwattsused = totalwattsused.astype(np.float)
			twu=np.sum(totalwattsused)
			billamount+=twu*D
			billamount+=(float(df[169+i][1])*D)
			flag=1
			ticks+=169
		else:
			billamount+=float(df[169+i][1])*D
			ticks+=1
		if ticks == (arr[p1]*24):
			ticks=0
			print('Bill for the month of {} is {}'.format(arr1[p1],str(billamount)))
			data.append(billamount)
			billamount=0.0
			p1+=1
			if p1==12:
				break
	print('Bill for the month of {} is {}'.format(arr1[p1],str(billamount)))
	data.append(billamount)
	return data

def staticpricingbill():
	S1=float(input('Enter Slab 1 margin:-'))
	S2=float(input('Enter Slab 2 margin:-'))
	S3=float(input('Enter Slab 3 margin:-'))
	P1=float(input('Enter Price of 1st band (0-S1):-'))
	P2=float(input('Enter Price of 2nd band (S1-S2):-'))
	P3=float(input('Enter Price of 3rd band (S2-S3):-'))
	P4=float(input('Enter Price of 4th band (S3 and above):-'))
	arr=[31,28,31,30,31,30,31,31,30,31,30,31]
	arr1=['January','Februray','March','April','May','June','July','August','September','October','November','December']
	df=pd.read_csv('sample1.csv')
	df=np.array(df)
	x=0
	data=[]
	print()
	for i in range(12):
		monthlybill=0.0
		twu=df[x:arr[i]*24+x,1]
		x+=arr[i]*24
		twu=twu.astype(np.float)
		totalwatts=np.sum(twu)
		if totalwatts <= S1:
			monthlybill=totalwatts*P1
		elif totalwatts>S1 and totalwatts<=S2:
			monthlybill=S1*P1+(totalwatts-S1)*P2
		elif totalwatts>S2 and totalwatts<=S3:
			monthlybill=S1*P1+S2*P2+(totalwatts-S1-S2)*P3
		else:
			monthlybill=S1*P1+S2*P2+S3*P3+(totalwatts-S1-S2-S3)*P4
		print('Bill for the month of {} is {}'.format(arr1[i],str(monthlybill)))
		data.append(monthlybill)
	return data
