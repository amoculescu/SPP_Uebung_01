//
// Created by Andrei Moculescu on 29.10.18.
//
#include <stdio.h>
#include <stdlib.h>
#include "sorted_list.h"

typedef struct SortedLinkedList {
    struct SortedLinkedListNode* first;
}SortedLinkedList;

typedef struct SortedLinkedListNode{
    SortedLinkedListNode* next;
    int data;
} SortedLinkedListNode;

SortedLinkedList* SortedLinkedList_create(){
    SortedLinkedList* list = malloc(sizeof(SortedLinkedList));
    list->first = NULL;
    return list;
}

void SortedLinkedList_addToList( SortedLinkedList* list, int data ){
    SortedLinkedListNode *new_entry = malloc(sizeof(SortedLinkedListNode));
	new_entry->next = NULL;
	new_entry->data = data;
	// If the list is empty -> Insert as new first
	if(list->first == NULL) {
		list->first = new_entry;
	// If the new value is smaller than the old first -> insert as the new first point
	}else if(data <= list->first->data){
	    new_entry->next = list->first;
	    list->first = new_entry;
	}else { 
        SortedLinkedListNode* current_entry = list->first;
		// go through the list until it either ende or the correct locatio is reached
		while(current_entry->next != NULL && current_entry->next->data <= data) {
			current_entry = current_entry->next;
		}
            new_entry->next = current_entry->next;
            current_entry->next = new_entry;
	}
}

void test(){
    int j = 1;
    int i;
    for(i = 0; i < 10; i++)
        j += 10;
    printf("j : %d", j);
}

SortedLinkedListNode* SortedLinkedList_getSmallest(SortedLinkedList* list){
    if(list->first == NULL){
        return NULL;
    }else{
        return list->first;
    }
}

void SortedLinkedList_delete(SortedLinkedList* list){
    SortedLinkedListNode* current_entry;
	while(list->first != NULL){
		current_entry = list->first;
		list->first = current_entry->next;
		free(current_entry);
	}
	free(list);
	// list = NULL;
}


SortedLinkedList* testCreateList(){
    SortedLinkedList* test_list = SortedLinkedList_create();
    printf("created a list at address %p\n", test_list);
    return test_list;
}

void testAddtoList(SortedLinkedList* test_list){
    int i;
    for(i = 0; i < 10; i++){
        int data = rand() % 11;
        printf("added %d to list\n", data);
        SortedLinkedList_addToList(test_list, data);
    }
    int done = 0;
    SortedLinkedListNode* currentElement = test_list->first;
    printf("going through list at %p again\n", test_list);
    while (!done){
        printf("Element in list found, address is %p, data is %d\n", currentElement, currentElement->data);
        if(currentElement->next != NULL){
            currentElement = currentElement->next;
        }else{
            printf("reached end of list\n");
            done = 1;
        }
    }
}

void testSmallestInList(SortedLinkedList* test_list){
    SortedLinkedListNode* result = SortedLinkedList_getSmallest(test_list);
    printf("smallest item in list is %d\n", result->data);
}

void testDelete(SortedLinkedList* test_list){
    SortedLinkedList_delete(test_list);
    if(test_list == NULL){
        printf("deletion was successful");
    }else {
        printf("deletion failed list is still at %p first with data, at address %p\n", test_list,test_list->first);
    }
}

void printList(SortedLinkedList* list){
    if (list->first != NULL) {
        SortedLinkedListNode *current = list->first;
        while (current->next != NULL) {
            printf("Item: %d\n", current->data);
            current = current->next;
        }
        printf("Item: %d\n", current->data);
    }
}

void addToListInteractive(SortedLinkedList* list){
    int c;
    do{
        printf("insert number, if you wish to exit insert -1\n");
        scanf("%d", &c);
        if(c != -1){
           printf("before:\n");
           printList(list);
           SortedLinkedList_addToList(list, c);
           printf("after:\n");
           printList(list);
        }
    } while(c != -1);
}

int main() {
    SortedLinkedList* test_list = testCreateList();
    testAddtoList(test_list);
    testSmallestInList(test_list);
    test();
    testDelete(test_list);

    test_list = testCreateList();
    addToListInteractive(test_list);
    return 0;
}
