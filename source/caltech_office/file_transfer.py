
f = open("/home/fyj/桌面/Multitask_learning/MTlearn-master/data/office/webcam/test_20.txt","r")
f_ = open("/home/fyj/桌面/Multitask_learning/MTlearn-master/data/office/webcam/test_20_.txt","w")
name = f.readline().split('_')

while name!=['']:
	str = ''
	name_ = str.join(['/home/fyj/桌面/Multitask_learning/dataset/webcam/',name[-3],'/',name[-2],'/',name[-1]])
	f_.write(name_)
	name = f.readline().split('/')
	print(name)
f.close()
f_.close()
