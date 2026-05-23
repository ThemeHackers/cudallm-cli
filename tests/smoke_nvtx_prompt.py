from src import profile_explain


def test_nvtx_and_prompt():
    sample = 'Some output... "Domain A@Range A" more text NVTX info\nTotal Time: 0.123 s'
    ranges = profile_explain.detect_nvtx_ranges(sample)
    assert ranges, 'NVTX ranges should be detected'
    combined = profile_explain.explain_combined(sample, None)
    assert 'generated_prompt' in combined
    print('NVTX SMOKE OK')


if __name__ == '__main__':
    test_nvtx_and_prompt()
