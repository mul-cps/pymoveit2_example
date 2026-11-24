from rclpy.parameter_client import AsyncParameterClient


# from ros2param.api.call_get_parameters
def node_get_parameter_str(*, node, executor=None, node_name, parameter_name):
    client = AsyncParameterClient(node, node_name)
    ready = client.wait_for_services(timeout_sec=5.0)
    if not ready:
        raise RuntimeError('Wait for service timed out waiting for '
                           f'parameter services for node {node_name}')
    future = client.get_parameters([parameter_name])

    if executor is None:
        while not future.done():
            pass
    else:
        executor.spin_until_future_complete(future, timeout_sec=None)

    if not future.done():
        raise RuntimeError(
            f'Timed out waiting for get_parameters response from node {node_name} '
            f'(timeout: {timeout}s)')
    response = future.result()
    if response is None:
        raise RuntimeError('Exception while calling service of node '
                           f'{node_name}: {future.exception()}')
    return response.values[0].string_value
