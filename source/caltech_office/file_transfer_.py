
f = open("/home/fyj/桌面/Multitask_learning/MTlearn-master/data/office/webcam/test_20.txt","r")
f_ = open("/home/fyj/桌面/Multitask_learning/MTlearn-master/data/office/webcam/test_20_.txt","w")
file = f.readline().split('/')[-1].split('_')
print(file)
print("yes")

while file!=['']:

	str = ''
	name = '/home/fyj/桌面/Multitask_learning/dataset/webcam/'
	num = len(file)
	if num == 4:
		name = str.join([name,file[-4],'_',file[-3],'/',file[-2],'_',file[-1]])

	elif num == 3:
		name = str.join([name,file[-3],'/',file[-2],'_',file[-1]])

	elif num == 2:
		name = str.join([name,file[-2],'/',file[-1]])

	f_.write(name)
	print(name)
	file = f.readline().split('/')[-1].split('_')

f.close()
f_.close()
