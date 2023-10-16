import re


def is_nonsensical_spec(spec: str, conj_exclusions: list[str]) -> bool:
    g_part: str = re.findall("G\((.*)\)", spec)[0]

    is_nonsensical: bool = False

    for exclusion in conj_exclusions:
        if exclusion not in g_part:
            is_nonsensical = False
        else:
            if "!" + exclusion in g_part:
                is_nonsensical = False
            else:
                parenthetical: list[str] = re.findall(f"\(.*\)", g_part)

                if len(parenthetical) != 0:
                    is_nonsensical = False
                else:
                    if "|" in g_part:
                        is_nonsensical = False
                    else:
                        is_nonsensical = True
                        break

    return is_nonsensical
