//
// Created by Andrei Moculescu on 29.10.18.
//
#include <sorted_list.h>
#include <stdio.h>
#include <stdlib.h>

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

void SortedLinkedList_addToList(SortedLinkedList* list, int data ){
    SortedLinkedListNode *new_entry = malloc(sizeof(SortedLinkedListNode));
    new_entry->next = NULL;
    new_entry->data = data;

    if(list->first == NULL) {
        list->first = new_entry;
    }else{
        SortedLinkedListNode* current_entry = list->first;
        //data <= first in list
        if(current_entry->data >= data){
            new_entry->next = current_entry;
            list->first = new_entry;
        }else {
            int found = 0;
            //look for first element bigger than data
            while ((current_entry->next != NULL) && (!found)) {
                if(current_entry->next->data >= data){
                    new_entry->next = current_entry->next;
                    current_entry->next = new_entry;
                    found = 1;
                }else{
                    current_entry = current_entry->next;
                }
            }
            //inserted element if bigger than all elements in list
            if(!found){
                current_entry->next = new_entry;
            }
        }
    }
}

SortedLinkedListNode* SortedLinkedList_getSmallest(SortedLinkedList* list){
    if(list->first == NULL){
        return NULL;
    }else{
        return list->first;
    }
}

void SortedLinkedList_delete(SortedLinkedList* list){
    int done = 0;
    SortedLinkedListNode* current_element;
    while(!done) {
        current_element = list->first;
        if (current_element->next != NULL) {
            list->first = current_element->next;
            free(current_element);
            current_element->next = NULL;
            current_element = NULL;
        } else {
            //free(current_element);
            done = 1;
        }
    }
    free(list->first);
    list->first = NULL;
    free(list);
    list = NULL;
}

SortedLinkedList* testCreateList(){
    SortedLinkedList* test_list = SortedLinkedList_create();
    printf("created a list at address %p\n", test_list);
    return test_list;
}

void testAddtoList(SortedLinkedList* test_list){
    for(int i = 0; i < 10; i++){
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

int main() {
    char c;
    SortedLinkedList* test_list = testCreateList();
    testAddtoList(test_list);
    testSmallestInList(test_list);
//    testDelete(test_list); broken if not on osx
    printf("insert number");

    c = getchar();

    printf("before:\n");
    printList(test_list);
    SortedLinkedList_addToList(test_list, (int) c - '0');
    printf("after:\n");
    printList(test_list);
    return 0;
}
