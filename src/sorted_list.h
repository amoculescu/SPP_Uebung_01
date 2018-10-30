#ifndef SORTED_LIST_H
#define SORTED_LIST_H

typedef struct SortedLinkedList SortedLinkedList;
typedef struct SortedLinkedListNode SortedLinkedListNode;

SortedLinkedList* 
SortedLinkedList_create();

void 
SortedLinkedList_addToList( SortedLinkedList* list, int data );

void
SortedLinkedList_delete( SortedLinkedList* list );

SortedLinkedListNode*
SortedLinkedList_getSmallest( SortedLinkedList* list );

#endif
