#ifndef WORLD_HPP
#define WORLD_HPP

#include "headers/math.hpp"
#include <stdint.h>

#define WORLD_SIZE 100
#define WORLD_HEIGHT 50

#define BLOCK_ID_AIR 0x00
#define BLOCK_ID_BEDROCK 0x01

#define BLOCK_ID_GRASS 0x02
#define BLOCK_ID_DIRT 0x03
#define BLOCK_ID_SAND 0x04
#define BLOCK_ID_GRAVEL 0x05

#define BLOCK_ID_STONE 0x06
#define BLOCK_ID_COBBLESTONE 0x07

#define BLOCK_ID_ORE_COAL 0x08
#define BLOCK_ID_ORE_IRON 0x09
#define BLOCK_ID_ORE_GOLD 0x0a
#define BLOCK_ID_ORE_DIAMOND 0x0b

#define BLOCK_ID_OAK_LOG 0x0c
#define BLOCK_ID_OAK_LEAVES 0x0d
#define BLOCK_ID_OAK_PLANKS 0x0e
#define BLOCK_ID_OAK_SLAB 0x0f
#define BLOCK_ID_OAK_STAIRS 0x10

#define BLOCK_ID_SPRUCE_LOG 0x11
#define BLOCK_ID_SPRUCE_LEAVES 0x12
#define BLOCK_ID_SPRUCE_PLANKS 0x13
#define BLOCK_ID_SPRUCE_SLAB 0x14
#define BLOCK_ID_SPRUCE_STAIRS 0x15

#define BLOCK_ID_BIRCH_LOG 0x16
#define BLOCK_ID_BIRCH_LEAVES 0x17
#define BLOCK_ID_BIRCH_PLANKS 0x18
#define BLOCK_ID_BIRCH_SLAB 0x19
#define BLOCK_ID_BIRCH_STAIRS 0x1a

#define BLOCK_ID_GLASS 0x1b

#define BLOCK_ID_INVALID 0xff

enum BlockType {
    BLOCKTYPE_SOLID = 0,
    BLOCKTYPE_TRANSPARENT = 1,
    BLOCKTYPE_SPECIAL = 2
};

struct BlockDescriptor {
    const wchar_t* name;
    BlockType type;
    uint32_t color;
};

class World {
public:
    World();
    ~World();
    uint8_t GetBlockID(Vector3i position);
    uint8_t *GetBlockIDs();
    BlockDescriptor GetBlockDescriptor(uint8_t blockID);
    BlockDescriptor *GetBlockDescriptors();
private:
    uint8_t blockIDs[WORLD_SIZE][WORLD_HEIGHT][WORLD_SIZE];
};

#endif
