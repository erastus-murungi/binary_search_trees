    def _fixup_case3(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        parent = hole.parent
        assert self.is_node(parent) and parent.is_3node
        left, middle, right = parent.left, parent.middle, parent.right

        # we have about 6 cases
        if hole is left:
            if middle.is_2node:
                # R----(287 : 830)
                #      L----hole -> (a)
                #      M----(466) -> (b, c)
                #      R----X
                # R----(830)
                #      M----(287: 466) -> (a, b, c)
                #      R----X
                middle.data.insert(0, parent.data.pop(0))
                self._maybe_move_child(hole, middle, at_start=True)
            else:
                assert right.is_2node
                # R----(303 : 785)
                #      L----hole -> a
                #      M----(374 : 700) -> (b, c, d)
                #      R----(980) -> (e, f)

                # R----(700)
                #      L----(303 : 374) -> (a, b, c)
                #      R----(785 : 980) -> (d, e, f)

                right.data.insert(0, parent.data.pop())
                middle.data.insert(0, parent.data.pop())
                parent.data.append(middle.data.pop())
                self._maybe_move_child(hole, middle, at_start=True)
                self._maybe_move_child(middle, right, at_start=True)
        elif hole is middle:
            if left.is_2node:
                # R----(317 : 754)
                #      L----(194) -> (a, b)
                #      M----hole -> (c)
                #      R----X
                # R----
                left.data.append(parent.data.pop(0))
                self._maybe_move_child(hole, left, at_start=False)
            else:
                assert right.is_2node
                # R----(269 : 376)
                #      L----X
                #      M----hole -> (a)
                #      R----(535) -> (b, c)
                right.data.insert(0, parent.data.pop())
                self._maybe_move_child(hole, right, at_start=True)
        else:
            assert hole is right
            if middle.is_2node:
                # R----(306 : 721)
                #      L---- X
                #      M----(504) -> (a, b)
                #      R----hole -> (c)

                # R----(306 : 721)
                #      L---- X
                #      R ----(504: 721) -> (a, b, c)
                middle.data.append(parent.data.pop())
                self._maybe_move_child(hole, middle, at_start=False)
            else:
                assert left.is_2node
                # R----(317 : 754)
                #      L----(200) -> (a, b)
                #      M----(350 : 541) -> (c, d, e)
                #      R----hole -> f

                # R----(350)
                #      L----(200: 317) -> (a, b, c)
                #      M----(541: 754) -> (d, e, f)

                middle.data.append(parent.data.pop())
                left.data.append(parent.data.pop())
                parent.data.append(middle.data.pop(0))
                self._maybe_move_child(middle, left, at_start=False, first_child=True)
                self._maybe_move_child(hole, middle, at_start=False)
        parent.remove_child(hole)
        return None


    def _fixup_case4(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        # a < 47 < b < 259 < c < 362 < d < 527 < e < 552 < f < 964 < g
        # R----(362 : 965)
        #      L----(47 : 259) => (a, b, c)
        #      M----(527 : 552) => (d, e, f)
        #      R---- 'hole' => (g)

        # R----(362 : 552)
        #      L----(47 : 259) => (a, b, c)
        #      M----(527) => (d, e)
        #      R---- (964) => (f, g)
        parent = hole.parent
        assert self.is_node(parent)
        left, middle, right = parent.left, parent.middle, parent.right

        if hole is parent.right:
            right.data.append(parent.data.pop())
            parent.data.append(middle.data.pop())
            self._maybe_move_child(middle, right, at_start=True, first_child=False)
        # a < 211 < b < 476 < c < 757 < d < 849 < e < 872 < f < 923 < g
        # R----(757 : 849)
        #      L----(211 : 476) => (a, b, c)
        #      M---- 'hole' => (d)
        #      R----(872 : 923) => (e, f, g)

        # R----(757 : 872)
        #      L---- (211 : 476) => (a, b, c)
        #      M---- (849) => (d, e)
        #      R---- (923) => (f, g)

        elif hole is parent.middle:
            middle.data.append(parent.data.pop())
            parent.data.append(right.data.pop(0))
            self._maybe_move_child(right, middle, at_start=False, first_child=True)
        # a < 172 < b < 275 < c < 409 < d < 678 < e < 684 < f < 926 < g
        # R----(172 : 678)
        #      L---- 'hole' => (a)
        #      M----(275 : 409) => (b, c, d)
        #      R----(684 : 926) => (e, f, g)

        # R----(275 : 678)
        #      L----(172) => (a, b)
        #      M----(409) => (c, d)
        #      R----(684 : 926) => (e, f, g)
        else:
            assert hole is parent.left
            left.data.append(parent.data.pop(0))
            parent.data.insert(0, middle.data.pop(0))
            self._maybe_move_child(middle, left, at_start=False, first_child=True)
            
        # seal the hole if it does not have any children
        if not hole.children:
            hole.children = self.sentinel()

        return None