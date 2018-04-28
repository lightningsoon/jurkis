# def we(a,b):
#     t=a
#     print('???')
#     while True:
#         print(t)
#         print(b)
#         yield None
#
#
# if __name__ == '__main__':
#     f=we(1,'fa')
#
#     next(f)
#     next(f)
#     next(we(3,9))
#     try:
#         next(f)
#     except StopIteration:
#         pass

# global b
# def f(a,b):
#     global x
#     x=a
#     print('!!')
# def f2():
#     global x,b
#     t=x
#     print(t)
#     print('b',b)
# if __name__ == '__main__':
#     f(1,2)
#     b=2
#     f2()