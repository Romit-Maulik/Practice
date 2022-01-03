#include "tf_utils.hpp"
#include "scope_guard.hpp"
#include <iostream>
#include <vector>

int main() {
  auto graph_ = tf_utils::LoadGraph("ML_LES.pb");
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph_); }; // Auto-delete on scope exit.
  if (graph_ == nullptr) {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  auto input_ph_ = TF_Output{TF_GraphOperationByName(graph_, "input_placeholder"), 0};
  if (input_ph_.oper == nullptr) {
    std::cout << "Can't init input_ph_" << std::endl;
    return 2;
  }

  const std::vector<std::int64_t> input_dims = {1, 9};
  const std::vector<float> input_vals = {4.948654692193851069e-05,-1.416845935576197153e-03,1.695804398322601982e-04,-4.909234209068177434e-05,7.200956380997814788e-04,-3.949331152012949186e-07,1.155548212380012041e-01,-1.447936297672789625e-05,-1.249577196433397854e-05,4.991843687885162174e-03};

  auto input_tensor = tf_utils::CreateTensor(TF_FLOAT, input_dims, input_vals);
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor); }; // Auto-delete on scope exit.

  auto output_ = TF_Output{TF_GraphOperationByName(graph_, "output_value/BiasAdd"), 0};
  if (output_.oper == nullptr) {
    std::cout << "Can't init output_" << std::endl;
    return 3;
  }

  TF_Tensor* output_tensor = nullptr;
  SCOPE_EXIT{ tf_utils::DeleteTensor(output_tensor); }; // Auto-delete on scope exit.

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
  auto options = TF_NewSessionOptions();
  auto sess = TF_NewSession(graph_, options, status);
  TF_DeleteSessionOptions(options);

  if (TF_GetCode(status) != TF_OK) {
    return 4;
  }

  TF_SessionRun(sess,
                nullptr, // Run options.
                &input_ph_, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                &output_, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );

  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error run session";
    return 5;
  }

  TF_CloseSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error close session";
    return 6;
  }

  TF_DeleteSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error delete session";
    return 7;
  }

  auto data = static_cast<float*>(TF_TensorData(output_tensor));

  std::cout << "Output vals: " << data[0] << std::endl;

  return 0;
}
