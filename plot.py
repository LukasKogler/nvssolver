import matplotlib.pyplot as plt
import pickle

Ls, kappa_S, kappa_A = pickle.load(open("Lkappa5.pickle", "rb"))

plt.subplot(1,2,1)
plt.title("kappa S")
plt.plot(Ls, kappa_S, label="kappa")
#plt.plot(Ls, [kappa_S[0] * Ls[0]**-2 * x**2 for x in Ls], "--", label="kappa_0*L**2")
plt.plot(Ls, [kappa_S[-1] * Ls[-1]**-2 * x**2 for x in Ls], "--", label="kappa_max*L**2")
plt.legend()
# plt.show()

plt.subplot(1,2,2)
plt.title("kappa A")
plt.plot(Ls, kappa_A, label="kappa")
#plt.plot(Ls, [kappa_A[0] * Ls[0]**-2 * x**2 for x in Ls], "--", label="kappa_0*L**2")
#plt.plot(Ls, [kappa_A[-1] * Ls[-1]**-2 * x**2 for x in Ls], "--", label="kappa_max*L**2")
plt.legend()
plt.show()
