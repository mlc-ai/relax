
# def test_mlp_blockbuilder():
#     layers, in_size, out_size, hidden_size, batch_size = 3, 5, 5, 5, 4

#     ty1 = rx.DynTensorType(ndim=1, dtype="float32")
#     ty2 = rx.DynTensorType(ndim=2, dtype="float32")

#     input_list = [rx.Var("x", [batch_size, in_size], ty2)]
#     w_list = [rx.Var("w_0", [in_size, hidden_size], ty2)] + \
#         [rx.Var("w_" + str(i + 1), [hidden_size, hidden_size], ty2) for i in range(layers - 2)] + \
#         [rx.Var("w_" + str(layers - 1), [hidden_size, out_size], ty2)]
#     b_list = [rx.Var("b_" + str(i), [hidden_size], ty1) for i in range(layers - 1)] + \
#         [rx.Var("b_" + str(layers - 1), [out_size], ty1)]
#     label_list = [rx.Var("y", [batch_size, out_size], ty2)]
#     args_list = input_list + w_list + b_list + label_list

#     bb = rx.BlockBuilder()
#     with bb.function("MLP", args_list):
#         with bb.dataflow():
#             current = input_list[0]
#             for i in range(layers):
#                 lv0 = bb.emit(R.nn.matmul(current, w_list[i]))
#                 lv1 = bb.emit(R.add(lv0, b_list[i]))
#                 current = bb.emit(R.nn.relu(lv1) if i < layers - 1 else lv1)
#             loss = bb.emit(R.nn.softmax_cross_entropy(current, label_list[0]))
#             gv0 = bb.emit_output(loss)
#         bb.emit_func_output(gv0)

#     Before = bb.get()
#     After = relax.transform.SimpleAD(Before.get_global_var("MLP"), args_list)(Before)
#     # Check numerical gradients equal
#     args = []
#     for arg in After["MLP_adjoint"].params[:-1]:
#         shape = [int(l) for l in arg.shape]
#         args.append(rand("float32", *shape))
#     label = np.random.rand(batch_size, out_size).astype(np.float32)
#     label /= label.sum(axis=1, keepdims=True)
#     args.append(tvm.nd.array(label))

#     _, grad = _execute_mod(After, "MLP_adjoint", *args)

#     def func(*inputs):
#         loss = _execute_mod(Before, "MLP", *[tvm.nd.array(i) for i in inputs])
#         return loss.numpy()
#     check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])
