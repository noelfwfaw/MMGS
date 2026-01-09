import json
from typing import Union, Optional
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import RectType


@AgentServer.custom_recognition("IsNumberGreaterThanZero")
class IsNumberGreaterThanZero(CustomRecognition):
    """
    使用OCR识别图像中的数字，并判断其是否大于0。

    参数格式:
    {
        "roi": [x, y, width, height]，识别区域，默认[0, 0, 0, 0]表示全屏
        "ocr_name": 使用的OCR识别器名称，默认"MyCustomOCR"
    }

    返回:
    若识别成功，返回AnalyzeResult，其中 detail 包含：
        - number: 识别出的数字
        - greater_than_zero: 是否大于0
    若识别失败或非数字，返回 None
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:

        try:
            params = json.loads(argv.custom_recognition_param)
            roi = params.get("roi", [0, 0, 0, 0])
            ocr_name = params.get("ocr_name", "MyCustomOCR")

            print(f"[INFO] 使用OCR识别模块: {ocr_name}")
            print(f"[INFO] 设置识别区域: {roi}")

            # 执行OCR识别
            ocr_result = context.run_recognition(
                ocr_name,
                argv.image,
                pipeline_override={ocr_name: {"roi": roi}},
            )

            if not ocr_result or not isinstance(ocr_result, dict):
                print(f"[WARN] OCR识别返回无效结果: {ocr_result}")
                return None

            text = ocr_result.get("text", "").strip()

            try:
                number = int(text)
            except ValueError:
                print(f"[WARN] OCR识别内容不是有效整数: '{text}'")
                return None

            result = {
                "number": number,
                "greater_than_zero": number > 0,
            }

            print(f"[INFO] 成功识别数字: {number}")
            print(f"[INFO] 是否大于0: {number > 0}")

            return CustomRecognition.AnalyzeResult(
                box=roi,
                detail=json.dumps(result),
            )

        except Exception as e:
            print(f"[ERROR] OCR分析过程中发生异常: {e}")
            return None


@AgentServer.custom_recognition("NumberComparison")
class NumberComparison(CustomRecognition):
    """
    使用OCR识别图像中的数字，并与指定值进行比较。

    只有当比较结果为真时才算识别成功，会执行后续动作；
    如果比较结果为假，则识别失败，不会执行后续动作。

    参数格式:
    {
        "roi": [x, y, width, height]，识别区域，默认[0, 0, 0, 0]表示全屏
        "ocr_name": 使用的OCR识别器名称，默认"MyCustomOCR"
        "compare_value": 要比较的值（数字），必填
        "operator": 比较操作符，可选值：">", "<", ">=", "<=", "==", "!="，默认">="
    }

    返回:
    若OCR识别成功且比较结果为真，返回AnalyzeResult，会执行后续动作，其中 detail 包含：
        - number: 识别出的数字
        - compare_value: 比较值
        - operator: 比较操作符
        - result: 比较结果（true）
        - description: 比较描述（如"5 >= 3"）
    若OCR识别失败、内容不是数字，或比较结果为假，返回 None，不会执行后续动作
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:

        try:
            params = json.loads(argv.custom_recognition_param)
            roi = params.get("roi", [0, 0, 0, 0])
            ocr_name = params.get("ocr_name", "MyCustomOCR")
            compare_value = params.get("compare_value")
            operator = params.get("operator", ">=")

            if compare_value is None:
                print("[ERROR] 必须指定 compare_value 参数")
                return None

            if not isinstance(compare_value, (int, float)):
                print(f"[ERROR] compare_value 必须是数字，当前值: {compare_value}")
                return None

            valid_operators = [">", "<", ">=", "<=", "==", "!="]
            if operator not in valid_operators:
                print(f"[ERROR] 无效的比较操作符: {operator}，支持的操作符: {valid_operators}")
                return None

            print(f"[INFO] 使用OCR识别模块: {ocr_name}")
            print(f"[INFO] 设置识别区域: {roi}")
            print(f"[INFO] 比较值: {compare_value}，操作符: {operator}")

            # 执行OCR识别
            ocr_result = context.run_recognition(
                ocr_name,
                argv.image,
                pipeline_override={ocr_name: {"roi": roi}},
            )

            if not ocr_result or not isinstance(ocr_result, dict):
                print(f"[WARN] OCR识别返回无效结果: {ocr_result}")
                return None

            text = ocr_result.get("text", "").strip()

            try:
                number = float(text)
                # 如果是整数形式，转换为整数
                if number.is_integer():
                    number = int(number)
            except ValueError:
                print(f"[WARN] OCR识别内容不是有效数字: '{text}'")
                return None

            # 执行比较
            if operator == ">":
                comparison_result = number > compare_value
            elif operator == "<":
                comparison_result = number < compare_value
            elif operator == ">=":
                comparison_result = number >= compare_value
            elif operator == "<=":
                comparison_result = number <= compare_value
            elif operator == "==":
                comparison_result = number == compare_value
            elif operator == "!=":
                comparison_result = number != compare_value

            # 生成比较描述
            description = f"{number} {operator} {compare_value}"

            print(f"[INFO] 成功识别数字: {number}")
            print(f"[INFO] 比较结果: {description} = {comparison_result}")

            # 如果比较结果为假，返回None（识别失败，不执行后续动作）
            if not comparison_result:
                print(f"[INFO] 比较条件不满足，识别失败")
                return None

            result = {
                "number": number,
                "compare_value": compare_value,
                "operator": operator,
                "result": comparison_result,
                "description": description,
            }

            return CustomRecognition.AnalyzeResult(
                box=roi,
                detail=json.dumps(result),
            )

        except Exception as e:
            print(f"[ERROR] 数字比较分析过程中发生异常: {e}")
            return None
