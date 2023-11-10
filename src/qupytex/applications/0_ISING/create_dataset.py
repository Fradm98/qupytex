from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists
import numpy as np
import argparse

# Create datasets from qs-mps package
# parameters

parser = argparse.ArgumentParser(prog="Create Dataset")
parser.add_argument("L", help="Spin chain length", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "h_f", help="Final value of h (external transverse field)", type=float
)
parser.add_argument(
    "path", help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'", type=str
)
parser.add_argument("chi", help="Simulated bond dimensions", type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)

args = parser.parse_args()

# take the path and precision to save files
if args.path == 'pc':
    path_drive = "G:/My Drive/projects/0_ISING"
elif args.path == 'mac':
    path_drive = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING"
elif args.path == 'marcos':
    path_drive = "/Users/fradm/Google Drive/My Drive/projects/0_ISING"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")


interval = np.linspace(args.h_i, args.h_f, args.npoints).tolist()
precision = get_precision((args.h_f - args.h_i)/args.npoints)

X = []
Y = []
print('find the rdms and assign labels...')
for h in interval:
    print(f'h: {h:.{precision}f}')
    d = 2
    chain = MPS(L=args.L, d=d, model=args.model, chi=args.chi, h=h)
    chain.load_sites(path=path_drive, precision=precision)
    d = chain.reduced_density_matrix(sites=[args.L // 2])
    if h < 1:
        y = 1
    else:
        y = -1
    X.append(d)
    Y.append(y)

save_list_of_lists(f"{path_drive}/results/dataset/X_1-rdms_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}", X)
np.savetxt(f"{path_drive}/results/dataset/Y_1-rdms_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}", Y)

# # define training sample and labels
# print('split in train and test...')
# interval = interval[::20]
# h1_tr = 0.13603603603603603
# h2_tr = 0.7126126126126126
# h3_tr = 1.1450450450450451
# h4_tr = 1.7216216216216216
# X_train = X[interval.index(h1_tr):interval.index(h2_tr)] + X[interval.index(h3_tr):interval.index(h4_tr)] 
# Y_train = Y[interval.index(h1_tr):interval.index(h2_tr)] + Y[interval.index(h3_tr):interval.index(h4_tr)] 
# X_test = X[:interval.index(h1_tr)] + X[interval.index(h2_tr):interval.index(h3_tr)] +  X[interval.index(h4_tr):]
# Y_test = Y[:interval.index(h1_tr)] + Y[interval.index(h2_tr):interval.index(h3_tr)] +  Y[interval.index(h4_tr):]

# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# plt.imshow(np.asarray(Y_train).reshape(1,len(Y_train)), cmap='seismic', aspect='auto')
# plt.show()

# def custom_kernel(matrix_1, matrix_2 = None):
#     if matrix_2 == None:
#         matrix_2 = matrix_1
#     K_tr = np.zeros((len(matrix_1),len(matrix_2)))
#     for i in range(len(matrix_1)):
#         for j in range(i, len(matrix_2)):
#             if j == i:
#                 K_tr[i,j] = 1
#             else:
#                 K_tr[i,j] = uhlmann_fidelity_1q(matrix_1[i],matrix_2[j])
#                 if j < len(matrix_1):
#                     K_tr[j,i] = K_tr[i,j]
#     return K_tr

# # compute kernels
# print('compute training kernel...')
# k_train = custom_kernel(X_train)
# print('compute testing kernel...')
# k_test = custom_kernel(X_test, X_train)
# plt.title('train')
# plt.imshow(k_train, cmap='viridis')
# plt.colorbar()
# plt.show()
# print(k_train.shape)
# plt.title('test')
# plt.imshow(k_test, cmap='viridis')
# plt.colorbar()
# plt.show()
# print(k_test.shape)

# # Step 2: Create an SVC model with the custom kernel
# svm_model = SVC(kernel='precomputed')
# print('fitting...')
# svm_model.fit(k_train, Y_train)
# print('find the score...')
# score_exp = svm_model.score(k_test, Y_test)
# print('%s uhlmann kernel classification test score: %0.2f' %('experimental',score_exp))


# print(svm_model.n_support_)
# print(svm_model.support_)

# interval = interval[::10]

# idx_ferro = svm_model.support_[0]
# idx_para = svm_model.support_[1]
# supp_ferro = X_train[idx_ferro]
# where_ferro = X.index(supp_ferro)
# lambda_ferro = interval[where_ferro]

# supp_para = X_train[idx_para]
# where_para = X.index(supp_para)
# lambda_para = interval[where_para]

# transition = (lambda_para - lambda_ferro)/2
# print(f"the predicted transition is at: {transition}")