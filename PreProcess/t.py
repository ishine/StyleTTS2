from pypinyin import pinyin, lazy_pinyin, Style

print([i for j in "hello world, this is cool".split() for i in (j, ' ')][:-1])