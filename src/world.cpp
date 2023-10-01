#include "headers/world.hpp"

BlockDescriptor blockDescriptors[256] = {
    { L"Air", BlockType::BLOCKTYPE_TRANSPARENT, 0x000000 },
    { L"Bedrock", BlockType::BLOCKTYPE_SOLID, 0x333333 },

    { L"Grass", BlockType::BLOCKTYPE_SOLID, 0x00ff00 },
    { L"Dirt", BlockType::BLOCKTYPE_SOLID, 0x8b4513 },
    { L"Sand", BlockType::BLOCKTYPE_SOLID, 0xffff00 },
    { L"Gravel", BlockType::BLOCKTYPE_SOLID, 0x808080 },

    { L"Stone", BlockType::BLOCKTYPE_SOLID, 0x808080 },
    { L"Cobblestone", BlockType::BLOCKTYPE_SOLID, 0x808080 },

    { L"Coal Ore", BlockType::BLOCKTYPE_SOLID, 0x000000 },
    { L"Iron Ore", BlockType::BLOCKTYPE_SOLID, 0x000000 },
    { L"Gold Ore", BlockType::BLOCKTYPE_SOLID, 0x000000 },
    { L"Diamond Ore", BlockType::BLOCKTYPE_SOLID, 0x000000 },

    { L"Oak Log", BlockType::BLOCKTYPE_SOLID, 0x8b4513 },
    { L"Oak Leaves", BlockType::BLOCKTYPE_TRANSPARENT, 0x00ff00 },
    { L"Oak Planks", BlockType::BLOCKTYPE_SOLID, 0x8b4513 },
    { L"Oak Slab", BlockType::BLOCKTYPE_SPECIAL, 0x8b4513 },
    { L"Oak Stairs", BlockType::BLOCKTYPE_SPECIAL, 0x8b4513 },

    { L"Spruce Log", BlockType::BLOCKTYPE_SOLID, 0x8b4513 },
    { L"Spruce Leaves", BlockType::BLOCKTYPE_TRANSPARENT, 0x00ff00 },
    { L"Spruce Planks", BlockType::BLOCKTYPE_SOLID, 0x8b4513 },
    { L"Spruce Slab", BlockType::BLOCKTYPE_SPECIAL, 0x8b4513 },
    { L"Spruce Stairs", BlockType::BLOCKTYPE_SPECIAL, 0x8b4513 },

    { L"Birch Log", BlockType::BLOCKTYPE_SOLID, 0x8b4513 },
    { L"Birch Leaves", BlockType::BLOCKTYPE_TRANSPARENT, 0x00ff00 },
    { L"Birch Planks", BlockType::BLOCKTYPE_SOLID, 0x8b4513 },
    { L"Birch Slab", BlockType::BLOCKTYPE_SPECIAL, 0x8b4513 },
    { L"Birch Stairs", BlockType::BLOCKTYPE_SPECIAL, 0x8b4513 },

    { L"Glass", BlockType::BLOCKTYPE_TRANSPARENT, 0x00ffff },

    { L"Invalid", BlockType::BLOCKTYPE_SOLID, 0xff0000 }
};

World::World() {
    // Y = 0
    this->blockIDs[0][0][0] = BLOCK_ID_BEDROCK; this->blockIDs[0][0][1] = BLOCK_ID_BEDROCK; this->blockIDs[0][0][2] = BLOCK_ID_BEDROCK; this->blockIDs[0][0][3] = BLOCK_ID_BEDROCK; this->blockIDs[0][0][4] = BLOCK_ID_BEDROCK;
    this->blockIDs[1][0][0] = BLOCK_ID_BEDROCK; this->blockIDs[1][0][1] = BLOCK_ID_BEDROCK; this->blockIDs[1][0][2] = BLOCK_ID_BEDROCK; this->blockIDs[1][0][3] = BLOCK_ID_BEDROCK; this->blockIDs[1][0][4] = BLOCK_ID_BEDROCK;
    this->blockIDs[2][0][0] = BLOCK_ID_BEDROCK; this->blockIDs[2][0][1] = BLOCK_ID_BEDROCK; this->blockIDs[2][0][2] = BLOCK_ID_BEDROCK; this->blockIDs[2][0][3] = BLOCK_ID_BEDROCK; this->blockIDs[2][0][4] = BLOCK_ID_BEDROCK;
    this->blockIDs[3][0][0] = BLOCK_ID_BEDROCK; this->blockIDs[3][0][1] = BLOCK_ID_BEDROCK; this->blockIDs[3][0][2] = BLOCK_ID_BEDROCK; this->blockIDs[3][0][3] = BLOCK_ID_BEDROCK; this->blockIDs[3][0][4] = BLOCK_ID_BEDROCK;
    this->blockIDs[4][0][0] = BLOCK_ID_BEDROCK; this->blockIDs[4][0][1] = BLOCK_ID_BEDROCK; this->blockIDs[4][0][2] = BLOCK_ID_BEDROCK; this->blockIDs[4][0][3] = BLOCK_ID_BEDROCK; this->blockIDs[4][0][4] = BLOCK_ID_BEDROCK;

    // Y = 1
    this->blockIDs[0][1][0] = BLOCK_ID_STONE; this->blockIDs[0][1][1] = BLOCK_ID_STONE; this->blockIDs[0][1][2] = BLOCK_ID_STONE; this->blockIDs[0][1][3] = BLOCK_ID_STONE; this->blockIDs[0][1][4] = BLOCK_ID_STONE;
    this->blockIDs[1][1][0] = BLOCK_ID_STONE; this->blockIDs[1][1][1] = BLOCK_ID_STONE; this->blockIDs[1][1][2] = BLOCK_ID_STONE; this->blockIDs[1][1][3] = BLOCK_ID_STONE; this->blockIDs[1][1][4] = BLOCK_ID_STONE;
    this->blockIDs[2][1][0] = BLOCK_ID_STONE; this->blockIDs[2][1][1] = BLOCK_ID_STONE; this->blockIDs[2][1][2] = BLOCK_ID_STONE; this->blockIDs[2][1][3] = BLOCK_ID_STONE; this->blockIDs[2][1][4] = BLOCK_ID_STONE;
    this->blockIDs[3][1][0] = BLOCK_ID_STONE; this->blockIDs[3][1][1] = BLOCK_ID_STONE; this->blockIDs[3][1][2] = BLOCK_ID_STONE; this->blockIDs[3][1][3] = BLOCK_ID_STONE; this->blockIDs[3][1][4] = BLOCK_ID_STONE;
    this->blockIDs[4][1][0] = BLOCK_ID_STONE; this->blockIDs[4][1][1] = BLOCK_ID_STONE; this->blockIDs[4][1][2] = BLOCK_ID_STONE; this->blockIDs[4][1][3] = BLOCK_ID_STONE; this->blockIDs[4][1][4] = BLOCK_ID_STONE;

    // Y = 2
    this->blockIDs[0][2][0] = BLOCK_ID_STONE; this->blockIDs[0][2][1] = BLOCK_ID_STONE; this->blockIDs[0][2][2] = BLOCK_ID_STONE; this->blockIDs[0][2][3] = BLOCK_ID_STONE; this->blockIDs[0][2][4] = BLOCK_ID_STONE;
    this->blockIDs[1][2][0] = BLOCK_ID_STONE; this->blockIDs[1][2][1] = BLOCK_ID_STONE; this->blockIDs[1][2][2] = BLOCK_ID_STONE; this->blockIDs[1][2][3] = BLOCK_ID_STONE; this->blockIDs[1][2][4] = BLOCK_ID_STONE;
    this->blockIDs[2][2][0] = BLOCK_ID_STONE; this->blockIDs[2][2][1] = BLOCK_ID_STONE; this->blockIDs[2][2][2] = BLOCK_ID_STONE; this->blockIDs[2][2][3] = BLOCK_ID_STONE; this->blockIDs[2][2][4] = BLOCK_ID_STONE;
    this->blockIDs[3][2][0] = BLOCK_ID_STONE; this->blockIDs[3][2][1] = BLOCK_ID_STONE; this->blockIDs[3][2][2] = BLOCK_ID_STONE; this->blockIDs[3][2][3] = BLOCK_ID_STONE; this->blockIDs[3][2][4] = BLOCK_ID_STONE;
    this->blockIDs[4][2][0] = BLOCK_ID_STONE; this->blockIDs[4][2][1] = BLOCK_ID_STONE; this->blockIDs[4][2][2] = BLOCK_ID_STONE; this->blockIDs[4][2][3] = BLOCK_ID_STONE; this->blockIDs[4][2][4] = BLOCK_ID_STONE;

    // Y = 3
    this->blockIDs[0][3][0] = BLOCK_ID_DIRT; this->blockIDs[0][3][1] = BLOCK_ID_DIRT; this->blockIDs[0][3][2] = BLOCK_ID_DIRT; this->blockIDs[0][3][3] = BLOCK_ID_DIRT; this->blockIDs[0][3][4] = BLOCK_ID_DIRT;
    this->blockIDs[1][3][0] = BLOCK_ID_DIRT; this->blockIDs[1][3][1] = BLOCK_ID_DIRT; this->blockIDs[1][3][2] = BLOCK_ID_DIRT; this->blockIDs[1][3][3] = BLOCK_ID_DIRT; this->blockIDs[1][3][4] = BLOCK_ID_DIRT;
    this->blockIDs[2][3][0] = BLOCK_ID_DIRT; this->blockIDs[2][3][1] = BLOCK_ID_DIRT; this->blockIDs[2][3][2] = BLOCK_ID_DIRT; this->blockIDs[2][3][3] = BLOCK_ID_DIRT; this->blockIDs[2][3][4] = BLOCK_ID_DIRT;
    this->blockIDs[3][3][0] = BLOCK_ID_DIRT; this->blockIDs[3][3][1] = BLOCK_ID_DIRT; this->blockIDs[3][3][2] = BLOCK_ID_DIRT; this->blockIDs[3][3][3] = BLOCK_ID_DIRT; this->blockIDs[3][3][4] = BLOCK_ID_DIRT;
    this->blockIDs[4][3][0] = BLOCK_ID_DIRT; this->blockIDs[4][3][1] = BLOCK_ID_DIRT; this->blockIDs[4][3][2] = BLOCK_ID_DIRT; this->blockIDs[4][3][3] = BLOCK_ID_DIRT; this->blockIDs[4][3][4] = BLOCK_ID_DIRT;

    // Y = 4
    this->blockIDs[0][4][0] = BLOCK_ID_DIRT; this->blockIDs[0][4][1] = BLOCK_ID_DIRT; this->blockIDs[0][4][2] = BLOCK_ID_DIRT; this->blockIDs[0][4][3] = BLOCK_ID_DIRT; this->blockIDs[0][4][4] = BLOCK_ID_DIRT;
    this->blockIDs[1][4][0] = BLOCK_ID_DIRT; this->blockIDs[1][4][1] = BLOCK_ID_GRASS; this->blockIDs[1][4][2] = BLOCK_ID_GRASS; this->blockIDs[1][4][3] = BLOCK_ID_DIRT; this->blockIDs[1][4][4] = BLOCK_ID_DIRT;
    this->blockIDs[2][4][0] = BLOCK_ID_DIRT; this->blockIDs[2][4][1] = BLOCK_ID_GRASS; this->blockIDs[2][4][2] = BLOCK_ID_GRASS; this->blockIDs[2][4][3] = BLOCK_ID_DIRT; this->blockIDs[2][4][4] = BLOCK_ID_DIRT;
    this->blockIDs[3][4][0] = BLOCK_ID_DIRT; this->blockIDs[3][4][1] = BLOCK_ID_GRASS; this->blockIDs[3][4][2] = BLOCK_ID_GRASS; this->blockIDs[3][4][3] = BLOCK_ID_GRASS; this->blockIDs[3][4][4] = BLOCK_ID_DIRT;
    this->blockIDs[4][4][0] = BLOCK_ID_DIRT; this->blockIDs[4][4][1] = BLOCK_ID_DIRT; this->blockIDs[4][4][2] = BLOCK_ID_DIRT; this->blockIDs[4][4][3] = BLOCK_ID_DIRT; this->blockIDs[4][4][4] = BLOCK_ID_DIRT;

    // Y = 5
    this->blockIDs[0][5][0] = BLOCK_ID_GRASS; this->blockIDs[0][5][1] = BLOCK_ID_GRASS; this->blockIDs[0][5][2] = BLOCK_ID_GRASS; this->blockIDs[0][5][3] = BLOCK_ID_GRASS; this->blockIDs[0][5][4] = BLOCK_ID_GRASS;
    this->blockIDs[1][5][0] = BLOCK_ID_GRASS; this->blockIDs[1][5][1] = BLOCK_ID_AIR; this->blockIDs[1][5][2] = BLOCK_ID_OAK_LOG; this->blockIDs[1][5][3] = BLOCK_ID_GRASS; this->blockIDs[1][5][4] = BLOCK_ID_GRASS;
    this->blockIDs[2][5][0] = BLOCK_ID_GRASS; this->blockIDs[2][5][1] = BLOCK_ID_AIR; this->blockIDs[2][5][2] = BLOCK_ID_AIR; this->blockIDs[2][5][3] = BLOCK_ID_GRASS; this->blockIDs[2][5][4] = BLOCK_ID_GRASS;
    this->blockIDs[3][5][0] = BLOCK_ID_GRASS; this->blockIDs[3][5][1] = BLOCK_ID_AIR; this->blockIDs[3][5][2] = BLOCK_ID_AIR; this->blockIDs[3][5][3] = BLOCK_ID_AIR; this->blockIDs[3][5][4] = BLOCK_ID_GRASS;
    this->blockIDs[4][5][0] = BLOCK_ID_GRASS; this->blockIDs[4][5][1] = BLOCK_ID_GRASS; this->blockIDs[4][5][2] = BLOCK_ID_GRASS; this->blockIDs[4][5][3] = BLOCK_ID_GRASS; this->blockIDs[4][5][4] = BLOCK_ID_GRASS;

    // Y = 6
    this->blockIDs[0][6][0] = BLOCK_ID_AIR; this->blockIDs[0][6][1] = BLOCK_ID_AIR; this->blockIDs[0][6][2] = BLOCK_ID_AIR; this->blockIDs[0][6][3] = BLOCK_ID_AIR; this->blockIDs[0][6][4] = BLOCK_ID_AIR;
    this->blockIDs[1][6][0] = BLOCK_ID_AIR; this->blockIDs[1][6][1] = BLOCK_ID_AIR; this->blockIDs[1][6][2] = BLOCK_ID_OAK_LOG; this->blockIDs[1][6][3] = BLOCK_ID_AIR; this->blockIDs[1][6][4] = BLOCK_ID_AIR;
    this->blockIDs[2][6][0] = BLOCK_ID_AIR; this->blockIDs[2][6][1] = BLOCK_ID_AIR; this->blockIDs[2][6][2] = BLOCK_ID_AIR; this->blockIDs[2][6][3] = BLOCK_ID_AIR; this->blockIDs[2][6][4] = BLOCK_ID_AIR;
    this->blockIDs[3][6][0] = BLOCK_ID_AIR; this->blockIDs[3][6][1] = BLOCK_ID_AIR; this->blockIDs[3][6][2] = BLOCK_ID_AIR; this->blockIDs[3][6][3] = BLOCK_ID_AIR; this->blockIDs[3][6][4] = BLOCK_ID_AIR;
    this->blockIDs[4][6][0] = BLOCK_ID_AIR; this->blockIDs[4][6][1] = BLOCK_ID_AIR; this->blockIDs[4][6][2] = BLOCK_ID_AIR; this->blockIDs[4][6][3] = BLOCK_ID_AIR; this->blockIDs[4][6][4] = BLOCK_ID_AIR;

    // Y = 7
    this->blockIDs[0][7][0] = BLOCK_ID_AIR; this->blockIDs[0][7][1] = BLOCK_ID_AIR; this->blockIDs[0][7][2] = BLOCK_ID_AIR; this->blockIDs[0][7][3] = BLOCK_ID_AIR; this->blockIDs[0][7][4] = BLOCK_ID_AIR;
    this->blockIDs[1][7][0] = BLOCK_ID_AIR; this->blockIDs[1][7][1] = BLOCK_ID_AIR; this->blockIDs[1][7][2] = BLOCK_ID_OAK_LOG; this->blockIDs[1][7][3] = BLOCK_ID_AIR; this->blockIDs[1][7][4] = BLOCK_ID_AIR;
    this->blockIDs[2][7][0] = BLOCK_ID_AIR; this->blockIDs[2][7][1] = BLOCK_ID_AIR; this->blockIDs[2][7][2] = BLOCK_ID_AIR; this->blockIDs[2][7][3] = BLOCK_ID_AIR; this->blockIDs[2][7][4] = BLOCK_ID_AIR;
    this->blockIDs[3][7][0] = BLOCK_ID_AIR; this->blockIDs[3][7][1] = BLOCK_ID_AIR; this->blockIDs[3][7][2] = BLOCK_ID_AIR; this->blockIDs[3][7][3] = BLOCK_ID_AIR; this->blockIDs[3][7][4] = BLOCK_ID_AIR;
    this->blockIDs[4][7][0] = BLOCK_ID_AIR; this->blockIDs[4][7][1] = BLOCK_ID_AIR; this->blockIDs[4][7][2] = BLOCK_ID_AIR; this->blockIDs[4][7][3] = BLOCK_ID_AIR; this->blockIDs[4][7][4] = BLOCK_ID_AIR;

    // Y = 8
    this->blockIDs[0][8][0] = BLOCK_ID_AIR; this->blockIDs[0][8][1] = BLOCK_ID_OAK_LEAVES; this->blockIDs[0][8][2] = BLOCK_ID_OAK_LEAVES; this->blockIDs[0][8][3] = BLOCK_ID_OAK_LEAVES; this->blockIDs[0][8][4] = BLOCK_ID_AIR;
    this->blockIDs[1][8][0] = BLOCK_ID_AIR; this->blockIDs[1][8][1] = BLOCK_ID_OAK_LEAVES; this->blockIDs[1][8][2] = BLOCK_ID_OAK_LEAVES; this->blockIDs[1][8][3] = BLOCK_ID_OAK_LEAVES; this->blockIDs[1][8][4] = BLOCK_ID_AIR;
    this->blockIDs[2][8][0] = BLOCK_ID_AIR; this->blockIDs[2][8][1] = BLOCK_ID_OAK_LEAVES; this->blockIDs[2][8][2] = BLOCK_ID_OAK_LEAVES; this->blockIDs[2][8][3] = BLOCK_ID_OAK_LEAVES; this->blockIDs[2][8][4] = BLOCK_ID_AIR;
    this->blockIDs[3][8][0] = BLOCK_ID_AIR; this->blockIDs[3][8][1] = BLOCK_ID_AIR; this->blockIDs[3][8][2] = BLOCK_ID_AIR; this->blockIDs[3][8][3] = BLOCK_ID_AIR; this->blockIDs[3][8][4] = BLOCK_ID_AIR;
    this->blockIDs[4][8][0] = BLOCK_ID_AIR; this->blockIDs[4][8][1] = BLOCK_ID_AIR; this->blockIDs[4][8][2] = BLOCK_ID_AIR; this->blockIDs[4][8][3] = BLOCK_ID_AIR; this->blockIDs[4][8][4] = BLOCK_ID_AIR;

    // Y = 9
    this->blockIDs[0][9][0] = BLOCK_ID_AIR; this->blockIDs[0][9][1] = BLOCK_ID_AIR; this->blockIDs[0][9][2] = BLOCK_ID_OAK_LEAVES; this->blockIDs[0][9][3] = BLOCK_ID_AIR; this->blockIDs[0][9][4] = BLOCK_ID_AIR;
    this->blockIDs[1][9][0] = BLOCK_ID_AIR; this->blockIDs[1][9][1] = BLOCK_ID_OAK_LEAVES; this->blockIDs[1][9][2] = BLOCK_ID_OAK_LEAVES; this->blockIDs[1][9][3] = BLOCK_ID_OAK_LEAVES; this->blockIDs[1][9][4] = BLOCK_ID_AIR;
    this->blockIDs[2][9][0] = BLOCK_ID_AIR; this->blockIDs[2][9][1] = BLOCK_ID_AIR; this->blockIDs[2][9][2] = BLOCK_ID_OAK_LEAVES; this->blockIDs[2][9][3] = BLOCK_ID_AIR; this->blockIDs[2][9][4] = BLOCK_ID_AIR;
    this->blockIDs[3][9][0] = BLOCK_ID_AIR; this->blockIDs[3][9][1] = BLOCK_ID_AIR; this->blockIDs[3][9][2] = BLOCK_ID_AIR; this->blockIDs[3][9][3] = BLOCK_ID_AIR; this->blockIDs[3][9][4] = BLOCK_ID_AIR;
    this->blockIDs[4][9][0] = BLOCK_ID_AIR; this->blockIDs[4][9][1] = BLOCK_ID_AIR; this->blockIDs[4][9][2] = BLOCK_ID_AIR; this->blockIDs[4][9][3] = BLOCK_ID_AIR; this->blockIDs[4][9][4] = BLOCK_ID_AIR;

    // Y = 10
    this->blockIDs[0][10][0] = BLOCK_ID_AIR; this->blockIDs[0][10][1] = BLOCK_ID_AIR; this->blockIDs[0][10][2] = BLOCK_ID_AIR; this->blockIDs[0][10][3] = BLOCK_ID_AIR; this->blockIDs[0][10][4] = BLOCK_ID_AIR;
    this->blockIDs[1][10][0] = BLOCK_ID_AIR; this->blockIDs[1][10][1] = BLOCK_ID_AIR; this->blockIDs[1][10][2] = BLOCK_ID_OAK_LEAVES; this->blockIDs[1][10][3] = BLOCK_ID_AIR; this->blockIDs[1][10][4] = BLOCK_ID_AIR;
    this->blockIDs[2][10][0] = BLOCK_ID_AIR; this->blockIDs[2][10][1] = BLOCK_ID_AIR; this->blockIDs[2][10][2] = BLOCK_ID_AIR; this->blockIDs[2][10][3] = BLOCK_ID_AIR; this->blockIDs[2][10][4] = BLOCK_ID_AIR;
    this->blockIDs[3][10][0] = BLOCK_ID_AIR; this->blockIDs[3][10][1] = BLOCK_ID_AIR; this->blockIDs[3][10][2] = BLOCK_ID_AIR; this->blockIDs[3][10][3] = BLOCK_ID_AIR; this->blockIDs[3][10][4] = BLOCK_ID_AIR;
    this->blockIDs[4][10][0] = BLOCK_ID_AIR; this->blockIDs[4][10][1] = BLOCK_ID_AIR; this->blockIDs[4][10][2] = BLOCK_ID_AIR; this->blockIDs[4][10][3] = BLOCK_ID_AIR; this->blockIDs[4][10][4] = BLOCK_ID_AIR;
}

World::~World() {}

uint8_t World::GetBlockID(Vector3i position) {
    if (position.x < 0 || position.x >= WORLD_SIZE || position.y < 0 || position.y >= WORLD_HEIGHT || position.z < 0 || position.z >= WORLD_SIZE) {
        return 0xff;
    }
    return this->blockIDs[position.x][position.y][position.z];
}

uint8_t* World::GetBlockIDs() {
    return (uint8_t*)this->blockIDs;
}

BlockDescriptor World::GetBlockDescriptor(uint8_t blockID) {
    return blockDescriptors[blockID];
}

BlockDescriptor* World::GetBlockDescriptors() {
    return blockDescriptors;
}
