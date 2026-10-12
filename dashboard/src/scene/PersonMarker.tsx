import { Html } from '@react-three/drei'
import type { RoomPerson } from './types'

/** Animated, read-only context marker for a person on the room plan. */
export function PersonMarker({ person, occupied }: {
  person: RoomPerson
  occupied?: boolean
}) {
  return (
    <group position={person.pos}>
      <mesh position={[0, 0.82, 0]}>
        <capsuleGeometry args={[0.2, 0.62, 4, 8]} />
        <meshStandardMaterial color={occupied ? '#c96f3f' : '#6f829c'} />
      </mesh>
      <mesh position={[0, 1.42, 0]}>
        <sphereGeometry args={[0.17, 16, 12]} />
        <meshStandardMaterial color="#e8b99a" />
      </mesh>
      <Html position={[0, 1.72, 0]} center distanceFactor={8}>
        <span style={{ color: '#f2f5fa', background: '#171c26',
          border: '1px solid #c96f3f', borderRadius: 12, padding: '2px 8px',
          whiteSpace: 'nowrap', fontSize: 11 }}>
          {person.name || person.person_id}
        </span>
      </Html>
    </group>
  )
}
