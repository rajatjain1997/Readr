import reader.reader as rd

session = rd.session()
print(rd.train(session, "./test_pics/8.jpg", 8))