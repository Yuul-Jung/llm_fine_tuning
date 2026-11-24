# 교재 전체가 함께 쓰는 모델 정보와 파일 내려받기
#
# 모델 Revision(커밋 해시)을 고정해, 저장소의 모델이 갱신되어도 교재 결과가 바뀌지 않게 한다.

from huggingface_hub import hf_hub_download

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def qwen_file(filename: str, local_only: bool = False) -> str:
    """Qwen2.5-0.5B-Instruct의 파일 하나를 내려받아(이미 있으면 재사용) 경로를 돌려준다.

    local_only=True이면 내려받지 않고, 캐시에 없을 때 오류를 낸다(테스트용).
    """
    return hf_hub_download(QWEN_ID, filename, revision=QWEN_REVISION,
                           local_files_only=local_only)
