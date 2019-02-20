#ifndef BAIDU_FEED_MLARCH_BABYLON_EXPECT_H
#define BAIDU_FEED_MLARCH_BABYLON_EXPECT_H

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

#endif  // BAIDU_FEED_MLARCH_BABYLON_EXPECT_H
